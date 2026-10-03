"""Selftest gate for the Mac (T4) day: run it first, and do not start the long measurement unless it passes.
It costs a few minutes of a 24 h block and protects the rest of it (Doc/implementation_plan_mac-m4.md §20.4, §22.6).

Checks: platform, disk and memory, Low Power Mode / thermal state, that the energy meter responds to load (and a
recorded powermetrics fixture exists), that llama.cpp runs CPU-only (Metal off), and, for each run configuration, that
F16, Q8_0 and Q4_0 give sane KL divergence against the stock path's F16 logits and plausible decode speed. A
configuration whose Q8_0 KLD is more than 200x the stock path's is dropped (wrong kernel, e.g. the SME path under
QEMU); an expected KleidiAI Q8_0 loss (8-40x) is recorded, not failed. Writes results/m4_selftest.json with the
usable configurations (feed them to scripts/m4_measure_table.py --configs) and the raw `pmset` text for fixtures.
Exit status 0 only if the run may proceed.

    PYTHONPATH=. python scripts/mac_selftest.py --model phi3-mini --build-kai build-kai --build-stock build-stock
Rehearsal elsewhere:  ... --allow-non-mac --meter mock --text <small text> --ctx 256
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from energy_meter import BUSY_CMD, get_meter
from evol_inference.dryrun_setup import find_sources
from evol_inference.genome import Precision
from evol_inference.eval_data import write_wikitext2_test
from evol_inference.llama_probes import parse_kld, parse_kleidiai_selection, parse_llama_bench_json, parse_perplexity
from evol_inference.mac_env import machine_state
from evol_inference.mac_measure import standard_configs
from evol_inference.model_spec import get_spec
from evol_inference.platform_info import PLATFORM_KEY, collect
from evol_inference.selftest_gates import (
    Check, gate_backend_is_cpu, gate_energy, gate_f16_floor, gate_machine_state, gate_platform, gate_q4, gate_q8,
    gate_resources, gate_speed, overall_ok, usable_configs,
)


def run(cmd, env=None) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=dict(os.environ) | (env or {}))
    if p.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed ({p.returncode}): {(p.stdout + p.stderr)[-300:]}")
    return p.stdout + "\n" + p.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--configs", nargs="*", default=["stock", "kai", "kai-nosme"], help="stock is required as the reference")
    ap.add_argument("--llama-dir", default=os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp"))
    ap.add_argument("--build-kai", default="build-kai")
    ap.add_argument("--build-stock", default="build-stock")
    ap.add_argument("--gguf-dir", default="models/gguf")
    ap.add_argument("--text", default=None)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--meter", default=None)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    ap.add_argument("--allow-non-mac", action="store_true", help="rehearsal on another host: the platform check passes with a note")
    ap.add_argument("--work-dir", default="/tmp/m4_selftest")
    ap.add_argument("--out", default="results/m4_selftest.json")
    a = ap.parse_args()

    spec = get_spec(a.model)
    llama = Path(a.llama_dir).expanduser()
    cfgs = {k: v for k, v in standard_configs(a.build_kai, a.build_stock).items() if k in a.configs}
    work = Path(a.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    sources = find_sources(spec, Path(a.gguf_dir))
    text = Path(a.text) if a.text else write_wikitext2_test(Path(a.gguf_dir) / "wikitext2_test.txt")
    checks: list[Check] = []
    data: dict = {"configs": {}}

    # --- environment ---
    checks.append(gate_platform(platform.system(), platform.machine(), a.allow_non_mac))
    info = collect(llama_cpp_dir=llama)
    need_gb = sum(p.stat().st_size for p in sources.values()) / 1e9 + 2
    checks += gate_resources(shutil.disk_usage(work).free / 1e9, info.get("memory_gib"), need_gb,
                             max(p.stat().st_size for p in sources.values()) / 2**30 * 1.3)
    st = machine_state()
    checks += gate_machine_state(st.ok, st.throttled, st.low_power_mode, st.detail)
    data["machine_raw"] = st.raw
    missing = [str(p) for c in cfgs.values() for p in (llama / c.build / "bin/llama-perplexity", llama / c.build / "bin/llama-bench") if not p.exists()]
    missing += [str(sources.get(p)) for p in (spec.reference_precision,) if p not in sources]
    missing += [str(p) for p in (Path(a.gguf_dir) / f"{spec.key}-{n}.gguf" for n in ("q8_0-pure", "q4_0-pure")) if not p.exists()]
    checks.append(Check("files-present", not missing, "fatal", "all binaries and GGUFs found" if not missing else f"missing: {missing}"))
    if missing or "stock" not in cfgs:
        if "stock" not in cfgs:
            checks.append(Check("stock-config", False, "fatal", "the stock reference configuration is required"))
        finish(a, checks, cfgs, data, info)
        return

    # --- energy meter ---
    meter = get_meter(a.meter)
    idle = meter.idle_baseline(a.idle_seconds)
    _, busy = meter.measure_cmd(BUSY_CMD)
    fixture = Path("pm_fixture.plist").exists()
    checks += gate_energy(meter.name, idle, busy.avg_power_w, fixture_present=fixture)
    data["energy"] = {"backend": meter.name, "idle_w": idle, "busy_w": busy.avg_power_w, "busy_s": busy.duration_s}

    f16, q8f, q4f = sources[Precision.FP16], Path(a.gguf_dir) / f"{spec.key}-q8_0-pure.gguf", Path(a.gguf_dir) / f"{spec.key}-q4_0-pure.gguf"
    base = work / "base_f16.kld"

    def ppl(c, gguf, *extra, ctx=None):
        return run([llama / c.build / "bin/llama-perplexity", "-m", gguf, "-f", text, "-c", ctx or a.ctx, "--chunks", 1, "-ngl", 0,
                    "-t", a.threads, *c.ppl_args, *extra], dict(c.env))

    # --- accuracy gates, per configuration, against the stock path's F16 logits ---
    stock = cfgs["stock"]
    ppl0 = parse_perplexity(ppl(stock, f16, "--kl-divergence-base", base))
    data["f16_ppl"] = ppl0
    kld: dict[str, dict] = {}
    for label, c in cfgs.items():
        k = {}
        try:
            for key, g in (("f16", f16), ("q8", q8f), ("q4", q4f)):
                k[key] = parse_kld(ppl(c, g, "--kl-divergence-base", base, "--kl-divergence")).mean_kld
            sel = {}
            for g in (q8f, q4f):
                sel |= parse_kleidiai_selection(ppl(c, g, "-v", ctx=32))
            data["configs"][label] = {"kld": k, "kleidiai": sel}
        except RuntimeError as e:
            checks.append(Check(f"runs[{label}]", False, "fatal", str(e)))
            continue
        kld[label] = k
    for label, k in kld.items():
        checks.append(gate_f16_floor(k["f16"], label))
        if label != "stock":
            checks += gate_q8(label, k["q8"], kld["stock"]["q8"])
            checks.append(gate_q4(label, k["q4"], kld["stock"]["q4"]))
    if "stock" in kld:
        checks += gate_q8("stock", kld["stock"]["q8"], kld["stock"]["q8"])

    # --- CPU-only and speed sanity ---
    tps = {}
    for label, c in cfgs.items():
        try:
            out = run([llama / c.build / "bin/llama-bench", "-m", q4f, "-p", 0, "-n", 32, "-r", 2, "-t", a.threads, "-ngl", 0, "-o", "json", *c.bench_args], dict(c.env))
            raw = json.loads(out[out.index("["): out.rindex("]") + 1])
            checks.append(gate_backend_is_cpu(str(raw[0].get("backends", "")), label))
            tps[label] = statistics.median(next(r for r in parse_llama_bench_json(out) if r.n_gen > 0).samples_ts)
        except (RuntimeError, ValueError, StopIteration) as e:
            checks.append(Check(f"bench[{label}]", False, "fatal", str(e)))
    for label, v in tps.items():
        data["configs"].setdefault(label, {})["decode_tps_q4"] = v
        if label != "stock" and "stock" in tps:
            checks.append(gate_speed(label, v, tps["stock"]))
    finish(a, checks, cfgs, data, info)


def finish(a, checks, cfgs, data, info) -> None:
    labels = list(cfgs)
    ok = overall_ok(checks, labels)
    usable = usable_configs(checks, labels) if ok else []
    print(f"\n{'check':28s} {'result':7s} detail")
    for c in checks:
        print(f"{c.name:28s} {'ok' if c.ok else ('FAIL' if c.severity == 'fatal' else 'warn'):7s} {c.detail}")
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}; usable configurations: {usable}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({PLATFORM_KEY: info, "args": vars(a), "ok": ok, "usable_configs": usable,
                                        "checks": [asdict(c) for c in checks], **data, "t": time.time()}, indent=2, default=str))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
