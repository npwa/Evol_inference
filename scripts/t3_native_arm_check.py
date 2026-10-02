"""Tier T3: native Arm (e.g. AWS Graviton) check of accuracy and speed, KleidiAI on vs off.
(Doc/implementation_plan_mac-m4.md §20, Doc/graviton_runbook.md)

For a model's F16 / Q8_0 / Q4_0 GGUFs and each configuration it records
  * accuracy: KL divergence vs the F16 model's logits (the project's accuracy metric), plus F16-vs-F16
    (implementation floor) -- this is where KleidiAI's per-row Q8_0 re-quantization shows up;
  * kernel path: `system_info` flags and KleidiAI's selected kernels (from a short verbose run);
  * speed: llama-bench prefill (pp) and decode (tg) tokens/s at several thread counts.
Configurations:  kai     = build with KleidiAI, default
                 kai-nr  = same build with --no-repack (disables KleidiAI at run time)
                 nokai   = build without KleidiAI (optional)
No power data: Graviton exposes no energy counter. Run on the instance; results are JSON with a
platform block (rejected by the report code without it).

    python scripts/t3_native_arm_check.py --model phi3-mini --build-kai build-kai --build-nokai build-nokai
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

from evol_inference.llama_probes import (
    parse_kld, parse_kleidiai_selection, parse_llama_bench_json, parse_perplexity, parse_system_info,
)
from evol_inference.platform_info import PLATFORM_KEY, collect

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp")).expanduser()
GGUF = Path(os.environ.get("GGUF_DIR", "models/gguf"))
FILES = (("f16", "f16"), ("q8", "q8_0-pure"), ("q4", "q4_0-pure"))


def run(cmd, env_extra=None) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=dict(os.environ) | (env_extra or {}))
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}): {(p.stdout + p.stderr)[-400:]}")
    return p.stdout + "\n" + p.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--build-kai", default="build-kai", help="llama.cpp build dir (under LLAMA_CPP_DIR) with KleidiAI")
    ap.add_argument("--build-nokai", default=None, help="optional build without KleidiAI")
    ap.add_argument("--text", default=str(GGUF / "wikitext2_test.txt"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, nargs="*", default=None, help="thread counts for llama-bench (default: 1, n/2, n)")
    ap.add_argument("--bench-reps", type=int, default=3)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = {k: GGUF / f"{a.model}-{n}.gguf" for k, n in FILES}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing GGUF files: {missing}")
    ncpu = os.cpu_count() or 1
    threads = a.threads or sorted({1, max(1, ncpu // 2), ncpu})
    # per-tool flags for "KleidiAI off at run time": llama-perplexity takes -nr, llama-bench takes --repack 0
    configs = [("kai", a.build_kai, [], []), ("kai-nr", a.build_kai, ["-nr"], ["--repack", "0"])]
    if a.build_nokai:
        configs.append(("nokai", a.build_nokai, [], []))
    work = Path(os.environ.get("T3_WORK", "/tmp/t3"))
    work.mkdir(parents=True, exist_ok=True)
    base = work / f"base_{a.model}_f16_c{a.ctx}.kld"
    exe = lambda b, n: LLAMA / b / "bin" / n  # noqa: E731

    result: dict = {PLATFORM_KEY: collect(llama_cpp_dir=LLAMA, extra={"host": socket.gethostname(), "model": a.model, "ctx": a.ctx}),
                    "model": a.model, "ctx": a.ctx, "threads": threads, "configs": {}}
    t0 = time.perf_counter()

    if not a.skip_accuracy:
        # base logits from the F16 model with the default (KleidiAI) build; F16 is not touched by KleidiAI's quantized kernels
        log = run([exe(a.build_kai, "llama-perplexity"), "-m", files["f16"], "-f", a.text, "-c", a.ctx, "--chunks", 1,
                   "-ngl", 0, "-t", ncpu, "--kl-divergence-base", base])
        result["ppl_f16"] = parse_perplexity(log)
        print(f"F16 perplexity {result['ppl_f16']:.4f}", flush=True)

    for name, build, extra, bench_extra in configs:
        r: dict = {"build": build, "extra_args": extra, "bench_extra_args": bench_extra}
        for f in ("q8", "q4"):
            log = run([exe(build, "llama-perplexity"), "-m", files[f], "-c", 32, "--chunks", 1, "-f", a.text, "-ngl", 0, "-t", 2, "-v", *extra])
            r.setdefault("system_info", parse_system_info(log))
            sel = parse_kleidiai_selection(log)
            if sel:
                r["kleidiai"] = sel
        r.setdefault("kleidiai", {})
        if not a.skip_accuracy:
            for f in ("f16", "q8", "q4"):
                log = run([exe(build, "llama-perplexity"), "-m", files[f], "-f", a.text, "-c", a.ctx, "--chunks", 1, "-ngl", 0,
                           "-t", ncpu, "--kl-divergence-base", base, "--kl-divergence", *extra])
                k = parse_kld(log)
                r[f"kld_{f}"], r[f"same_top_p_{f}"] = k.mean_kld, k.same_top_p
            print(f"  {name:7s} KLD  f16 {r['kld_f16']:.6f}  q8 {r['kld_q8']:.6f}  q4 {r['kld_q4']:.6f}   kernels {r['kleidiai']}", flush=True)
        if not a.skip_bench:
            r["bench"] = {}
            for f in ("f16", "q8", "q4"):
                for t in threads:
                    out = run([exe(build, "llama-bench"), "-m", files[f], "-p", 512, "-n", 128, "-r", a.bench_reps, "-t", t, "-ngl", 0, "-o", "json", *bench_extra])
                    rows = parse_llama_bench_json(out)
                    pp = next(x for x in rows if x.n_prompt > 0)
                    tg = next(x for x in rows if x.n_gen > 0)
                    r["bench"][f"{f}_t{t}"] = {"pp512_tps": statistics.median(pp.samples_ts), "tg128_tps": statistics.median(tg.samples_ts),
                                               "tg128_stddev": tg.stddev_ts}
                    print(f"  {name:7s} {f:3s} t={t:<2d} pp512 {r['bench'][f'{f}_t{t}']['pp512_tps']:8.1f}  tg128 {r['bench'][f'{f}_t{t}']['tg128_tps']:7.2f} tok/s", flush=True)
        result["configs"][name] = r

    result["wall_s"] = round(time.perf_counter() - t0, 1)
    out = Path(a.out or f"results/t3_native_{a.model}_{socket.gethostname()}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out} ({result['wall_s']} s)")


if __name__ == "__main__":
    main()
