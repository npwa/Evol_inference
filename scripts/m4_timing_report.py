"""Turn a rehearsal's measured table into per-evaluation timings and Mac-day projections (replaces the placeholder
estimates in the default queue; Doc/implementation_plan_mac-m4.md §23).

Per (genome, configuration) a row records wall_s (KL run + speed/energy run, plus cooldown waits) and bench_wall_s. This
reports the medians for quantized genomes (genomes with F16 blocks are reported separately: they are several times
slower), then projects how many hours it takes to measure the whole genome space of the target models for a machine
that is `speedup` times faster than the one measured, and how many genomes fit in the block.

    python scripts/m4_timing_report.py --table results/rehearsal_table_phi3-mini.jsonl --out results/rehearsal_timings.json
"""

import argparse
import json
import statistics
from pathlib import Path

from evol_inference.mac_measure import load_rows

# Cost of one evaluation relative to Phi-3-mini: weight bytes (Q8_0 file 8.5 GB vs 4.1 GB, Q4_0 4.5 vs 2.15) and thus both
# the 2048-token KL run and the speed run scale by about this factor on a bandwidth/compute-bound CPU.
MODEL_SCALE = {"phi3-mini": 1.0, "llama3.1-8b": 2.1}
GENOMES_2LEVEL = 256   # 2^8 genomes over {Q8_0, Q4_0}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and "wall_s" in r]
    out: dict = {"model": ok[0]["model"] if ok else None, "rows": len(ok), "configs": {}}
    for c in sorted({r["config"] for r in ok}):
        mine = [r for r in ok if r["config"] == c]
        quant = [r for r in mine if "F" not in r["genome"]]
        f16 = [r for r in mine if "F" in r["genome"]]
        entry = {"n_quantized": len(quant), "n_with_f16": len(f16)}
        if quant:
            wall = [r["wall_s"] for r in quant]
            bench = [r["bench_wall_s"] for r in quant if "bench_wall_s" in r]
            entry.update(median_wall_s=statistics.median(wall), max_wall_s=max(wall),
                         median_bench_s=statistics.median(bench) if bench else None,
                         median_kld_s=statistics.median(r["wall_s"] - r.get("bench_wall_s", 0) for r in quant))
        if f16:
            entry["median_wall_s_with_f16"] = statistics.median(r["wall_s"] for r in f16)
            entry["f16_genomes"] = sorted(r["genome"] for r in f16)
        out["configs"][c] = entry
    return out


def project(summary: dict, hours: float = 23.0, overhead_h: float = 1.5, speedups=(1.0, 1.5, 2.0, 3.0)) -> dict:
    """Hours to measure the full 2-level space (256 genomes) of each target model in every measured configuration, and the
    number of genomes that fit in `hours` minus `overhead_h` (selftest, scan, F16 reference), per speedup."""
    per_genome = sum(e["median_wall_s"] for e in summary["configs"].values() if "median_wall_s" in e)  # one genome, all configs
    proj = {"per_genome_all_configs_s": per_genome, "budget_h": hours, "overhead_h": overhead_h, "models": {}}
    for m, scale in MODEL_SCALE.items():
        proj["models"][m] = {
            f"x{s:g}": {"full_space_h": GENOMES_2LEVEL * per_genome * scale / s / 3600,
                        "genomes_in_budget": int((hours - overhead_h) * 3600 * s / (per_genome * scale)) if per_genome else 0}
            for s in speedups}
    return proj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--hours", type=float, default=23.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    s = summarize(load_rows(a.table))
    if not s["configs"]:
        raise SystemExit("no usable rows")
    p = project(s, a.hours)
    print(f"{s['model']}: {s['rows']} measured rows")
    for c, e in s["configs"].items():
        line = f"  {c:10s} quantized genomes: n={e['n_quantized']}"
        if "median_wall_s" in e:
            line += (f", median {e['median_wall_s']:.0f} s per (genome, config) = KL {e['median_kld_s']:.0f} s + speed/energy "
                     f"{e['median_bench_s'] or 0:.0f} s (max {e['max_wall_s']:.0f} s)")
        if "median_wall_s_with_f16" in e:
            line += f";  with F16 blocks ({e['n_with_f16']}): median {e['median_wall_s_with_f16']:.0f} s"
        print(line)
    print(f"\nall configurations together: {p['per_genome_all_configs_s']:.0f} s per genome (assembly included once)")
    print(f"\nfull 2-level space (256 genomes) and genomes that fit in {a.hours:g} h minus {p['overhead_h']:g} h overhead,"
          f" by how much faster the Mac is than this machine:")
    for m, d in p["models"].items():
        print(f"  {m:12s} " + "   ".join(f"x{k[1:]}: {v['full_space_h']:5.1f} h / {v['genomes_in_budget']:4d} genomes" for k, v in d.items()))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": s, "projection": p}, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
