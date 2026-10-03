"""Thread-count scan (plan decision D3): uniform genomes at each quantized precision, decode/prefill speed and energy per
token at 1..N threads, for each run configuration. Fixes the thread count for the search and shows the P-core / E-core
effect on macOS, where threads cannot be pinned and only the count is controlled. No accuracy measurement.

    PYTHONPATH=. python scripts/m4_thread_scan.py --model phi3-mini --configs stock kai --threads 1 2 4 6 8 10
"""

import argparse
import json
import os
import time
from pathlib import Path

from energy_meter import get_meter
from evol_inference.dryrun_setup import find_sources
from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.mac_measure import GenomeMeasurer, MeasureSettings, standard_configs
from evol_inference.model_spec import get_spec
from evol_inference.platform_info import PLATFORM_KEY, collect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--configs", nargs="*", default=["stock", "kai"])
    ap.add_argument("--precisions", nargs="*", default=["int8", "int4"], choices=["fp16", "int8", "int4"])
    ap.add_argument("--threads", type=int, nargs="*", default=None, help="default: 1, 2, 4, ... up to the CPU count")
    ap.add_argument("--llama-dir", default=os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp"))
    ap.add_argument("--build-kai", default="build-kai")
    ap.add_argument("--build-stock", default="build-stock")
    ap.add_argument("--gguf-dir", default="models/gguf")
    ap.add_argument("--work-dir", default="/tmp/m4_thread_scan")
    ap.add_argument("--meter", default=None)
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cooldown-wait", type=float, default=300.0)
    ap.add_argument("--out", default="results/m4_thread_scan.json")
    a = ap.parse_args()

    n = os.cpu_count() or 1
    threads = a.threads or sorted({t for t in (1, 2, 4, 6, 8, 10, n) if t <= n})
    spec = get_spec(a.model)
    cfgs = standard_configs(a.build_kai, a.build_stock)
    chosen = [cfgs[c] for c in a.configs]
    meter = get_meter(a.meter)
    m = GenomeMeasurer(spec, find_sources(spec, Path(a.gguf_dir)), Path(a.llama_dir).expanduser(), chosen, Path(a.gguf_dir) / "x.txt",
                       Path(a.work_dir), meter, MeasureSettings(threads=threads[0], n_prompt=a.n_prompt, n_gen=a.n_gen, repeats=a.repeats,
                                                                cooldown_wait_s=a.cooldown_wait), idle_s=a.idle_seconds)
    rows = []
    for label in a.configs:
        for p in a.precisions:
            genome = [Precision(p)] * N_SUPER_BLOCKS
            for t in threads:
                probe = m.bench[label]
                probe.threads = t
                se = probe.measure(genome)
                row = {"config": label, "precision": p, "threads": t, "decode_tps": se.decode_tps, "prefill_tps": se.prefill_tps,
                       "joules_per_token": se.joules_per_token, "avg_power_w": se.avg_power_w, "decode_spread": se.decode_tps_spread,
                       "synthetic": se.synthetic, "energy_backend": meter.name, "idle_w": m.idle_w}
                rows.append(row)
                print(f"{label:6s} {p:5s} t={t:<2d} decode {se.decode_tps:7.2f} tok/s  prefill {se.prefill_tps:8.1f}  "
                      f"{se.joules_per_token:.4f} J/tok  {se.avg_power_w:5.1f} W", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({PLATFORM_KEY: collect(), "model": spec.key, "threads": threads, "t": time.time(), "rows": rows}, indent=2))
    best = max((r for r in rows if r["config"] == a.configs[0]), key=lambda r: r["decode_tps"], default=None)
    if best:
        print(f"fastest decode for {a.configs[0]}: {best['precision']} at {best['threads']} threads ({best['decode_tps']:.1f} tok/s)")


if __name__ == "__main__":
    main()
