"""Tier T2: do the aarch64 builds of llama.cpp (cross-compiled, run under QEMU user-mode) agree
numerically with the x86 build, and which Arm kernel paths run? (Doc/implementation_plan_mac-m4.md §20)

Correctness only -- QEMU timings mean nothing. For each (build, emulated CPU) it measures, on the
small proxy model, with KL divergence (the project's accuracy metric):
  impl_f16   KLD of the aarch64 F16 model vs the x86 F16 model's logits   (pure implementation numerics)
  acc_q8     KLD of aarch64 Q8_0 vs the x86 F16 logits                    (accuracy incl. implementation)
  acc_q4     KLD of aarch64 Q4_0 vs the x86 F16 logits
  impl_q4    KLD of aarch64 Q4_0 vs the x86 *Q4_0* logits                 (implementation numerics, quantized)
and records `system_info` flags and KleidiAI's kernel selection from the startup log.

    PYTHONPATH=. .venv/bin/python scripts/t2_arm_emulated_check.py
"""

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evol_inference.llama_probes import parse_kld, parse_kleidiai_selection, parse_perplexity, parse_system_info
from evol_inference.platform_info import PLATFORM_KEY, collect

LLAMA = Path("~/work/llama.cpp").expanduser()
G = Path("models/gguf")
MODEL = "smollm2-360m"
TEXT = G / "wikitext2_head.txt"

# label, build dir, QEMU cpu model (None = native x86), what the run is meant to exercise, extra env
CONFIGS = [
    ("x86 (native, AVX512)", "build-cpu", None, "reference", {}),
    ("a64 generic NEON, no KleidiAI", "build-aarch64-kaiOFF", "max", "baseline armv8-a code", {}),
    ("a64 generic + KleidiAI, n1", "build-aarch64-kaiON", "neoverse-n1", "KleidiAI runtime pick on a dotprod-only CPU", {}),
    ("a64 generic + KleidiAI, max", "build-aarch64-kaiON", "max", "KleidiAI runtime pick on SVE/SVE2/I8MM/SME CPU", {}),
    ("a64 generic + KleidiAI, max, SME off", "build-aarch64-kaiON", "max", "same, GGML_KLEIDIAI_SME=0 (diagnostic override)",
     {"GGML_KLEIDIAI_SME": "0"}),
    ("a64 +i8mm, no KleidiAI, v1", "build-aarch64-i8mm", "neoverse-v1", "ggml SDOT/SMMLA kernels", {}),
    ("a64 +i8mm + KleidiAI, v1", "build-aarch64-i8mm-kai", "neoverse-v1", "KleidiAI I8MM/SVE kernels (Graviton3-like)", {}),
    ("a64 +sve, no KleidiAI, v1", "build-aarch64-sve", "neoverse-v1", "ggml SVE kernels", {}),
]


def run(build: str, cpu: str | None, gguf: Path, ctx: int, threads: int, *extra: str, env_extra: dict | None = None) -> str:
    env = dict(os.environ) | (env_extra or {})
    if cpu:
        env["QEMU_CPU"] = cpu
    cmd = [str(LLAMA / build / "bin/llama-perplexity"), "-m", str(gguf), "-f", str(TEXT), "-c", str(ctx),
           "--chunks", "1", "-ngl", "0", "-t", str(threads), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = p.stdout + "\n" + p.stderr
    if p.returncode != 0:
        raise RuntimeError(f"{build}/{cpu} {gguf.name} failed ({p.returncode}): {out[-400:]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--out", default="results/t2_arm_emulated.json")
    ap.add_argument("--only", default="", help="run only configs whose label contains this text; results are merged into --out")
    ap.add_argument("--attribution-only", action="store_true", help="only (re)capture system_info + KleidiAI kernel selection")
    a = ap.parse_args()
    files = {k: G / f"{MODEL}-{n}.gguf" for k, n in (("f16", "f16"), ("q8", "q8_0-pure"), ("q4", "q4_0-pure"))}
    work = Path("/dev/shm/t2")
    work.mkdir(exist_ok=True)
    base_f16, base_q4 = work / "x86_f16.kld", work / "x86_q4.kld"

    # x86 reference logits
    x86 = CONFIGS[0][1]
    ppl_f16 = parse_perplexity(run(x86, None, files["f16"], a.ctx, 8, "--kl-divergence-base", str(base_f16)))
    run(x86, None, files["q4"], a.ctx, 8, "--kl-divergence-base", str(base_q4))
    print(f"x86 reference: F16 ppl {ppl_f16:.4f} (ctx {a.ctx}, 1 chunk, {TEXT.name})", flush=True)

    def attribution(build, cpu, env_extra) -> dict:
        """Kernel-path attribution: a short VERBOSE run on the Q8_0 and Q4_0 files (KleidiAI only logs its
        kernel selection with -v, and only once it sees quantized weights)."""
        out = {}
        for f in ("q8", "q4"):
            log = run(build, cpu, files[f], 32, a.threads, "-v", env_extra=env_extra)
            out.setdefault("system_info", parse_system_info(log))
            sel = parse_kleidiai_selection(log)
            if sel:
                out["kleidiai"] = sel
        out.setdefault("kleidiai", {})
        return out

    def measure(cfg) -> dict:
        label, build, cpu, note, env_extra = cfg
        t0 = time.perf_counter()
        r: dict = {"label": label, "build": build, "qemu_cpu": cpu, "note": note, "env": env_extra}
        kld = lambda f, base: parse_kld(run(build, cpu, files[f], a.ctx, a.threads, "--kl-divergence-base", str(base), "--kl-divergence", env_extra=env_extra))  # noqa: E731
        try:
            r |= attribution(build, cpu, env_extra)
            if not a.attribution_only:
                r["impl_f16"] = kld("f16", base_f16).mean_kld
                r["acc_q8"] = kld("q8", base_f16).mean_kld
                q4 = kld("q4", base_f16)
                r["acc_q4"], r["acc_q4_same_top_p"] = q4.mean_kld, q4.same_top_p
                r["impl_q4"] = kld("q4", base_q4).mean_kld
        except RuntimeError as e:  # e.g. SIGILL when a build's instructions exceed the emulated CPU
            r["error"] = str(e)[:300]
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        print(f"  done {label}: {r.get('error') or 'ok'} ({r['wall_s']}s)", flush=True)
        return r

    todo = [c for c in CONFIGS if a.only in c[0]]
    with ThreadPoolExecutor(a.parallel) as ex:
        fresh = list(ex.map(measure, todo))
    # merge into any existing results by label (keeps earlier measurements; fresh values win key by key)
    old = {}
    if Path(a.out).exists():
        old = {r["label"]: r for r in json.loads(Path(a.out).read_text()).get("results", [])}
    for r in fresh:
        old[r["label"]] = old.get(r["label"], {}) | r
    results = [old[c[0]] for c in CONFIGS if c[0] in old]

    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps({PLATFORM_KEY: collect(), "model": MODEL, "ctx": a.ctx, "text": TEXT.name,
                                       "x86_f16_ppl": ppl_f16, "results": results}, indent=2))
    print(f"\n{'config':38s} {'impl_f16':>9s} {'acc_q8':>9s} {'acc_q4':>9s} {'impl_q4':>9s}  KleidiAI kernels / flags")
    for r in results:
        if "impl_f16" not in r:
            continue
        if "error" in r:
            print(f"{r['label']:38s} ERROR {r['error'][:100]}")
            continue
        kai = ",".join(f"{k}:{v}" for k, v in r["kleidiai"].items()) or "-"
        flags = ",".join(k for k in r["system_info"] if k in ("NEON", "MATMUL_INT8", "DOTPROD", "SVE", "SME", "KLEIDIAI"))
        print(f"{r['label']:38s} {r['impl_f16']:9.6f} {r['acc_q8']:9.6f} {r['acc_q4']:9.6f} {r['impl_q4']:9.6f}  {kai}  [{flags}]")


if __name__ == "__main__":
    main()
