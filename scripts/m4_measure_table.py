"""Measure every genome of a model's alphabet (or as many as the time budget allows) under one or more llama.cpp
run configurations, and append one JSON line per (genome, configuration) to a resumable table
(Doc/implementation_plan_mac-m4.md §9, §22.6). Used on the Mac (T4); rehearsed on the desktop and on Graviton.

  * accuracy: KL divergence vs the original model's logits; speed: llama-bench prefill/decode; energy: joules per
    decoded token from the energy meter (powermetrics on the Mac; mock/RAPL elsewhere, stamped synthetic if mock)
  * configurations are measured genome by genome, alternating order (A,B then B,A), so drift and thermal state do
    not bias one configuration; the assembled GGUF is shared between them
  * order: baselines, sensitivity-greedy chain, then the rest in seeded random order (any prefix is a random sample)
  * stops cleanly at --budget-seconds, when --stop-file appears, or after --limit genomes; rerun to resume

    PYTHONPATH=. python scripts/m4_measure_table.py --model phi3-mini --build-kai build-kai --build-stock build-stock
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from energy_meter import get_meter
from evol_inference.dryrun_setup import find_sources
from evol_inference.eval_data import write_wikitext2_test
from evol_inference.genome import Precision
from evol_inference.mac_measure import (
    GenomeMeasurer, MeasureSettings, append_row, assembly_space_gb, done_keys, load_rows, priority_order, standard_configs,
)
from evol_inference.model_spec import get_spec
from evol_inference.platform_info import PLATFORM_KEY, collect
from evol_inference.sensitivity import SensitivityTable

PREC = {p.value: p for p in Precision}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--alphabet", nargs="*", choices=list(PREC), default=None, help="default: the model's alphabet")
    ap.add_argument("--configs", nargs="*", default=["kai", "stock"], help="kai, kai-nosme, kai-nr, stock")
    ap.add_argument("--llama-dir", default=os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp"))
    ap.add_argument("--build-kai", default="build-kai")
    ap.add_argument("--build-stock", default="build-stock")
    ap.add_argument("--gguf-dir", default="models/gguf")
    ap.add_argument("--text", default=None, help="evaluation text (default: gguf-dir/wikitext2_test.txt, written if missing)")
    ap.add_argument("--work-dir", default=None, help="assembled GGUF + base logits (default: work/m4_<model> on disk; use a RAM disk on the Mac if you like; /tmp is RAM-backed on recent Ubuntu)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sensitivity", default=None, help="sensitivity table JSON for the greedy chain (default: results/sensitivity_<model>_kld.json if present)")
    ap.add_argument("--meter", default=None, help="energy backend: powermetrics, rapl, mock, replay (default: auto)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--n-prompt", type=int, default=512)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--energy-method", choices=["total", "differential"], default="total")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--idle-refresh-every", type=int, default=25, help="re-measure idle power every N genomes")
    ap.add_argument("--cooldown-wait", type=float, default=300.0, help="max seconds to wait out thermal throttling per measurement")
    ap.add_argument("--budget-seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="max genomes this run")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    spec = get_spec(a.model)
    alphabet = [PREC[x] for x in a.alphabet] if a.alphabet else list(spec.alphabet)
    out = Path(a.out or f"results/m4_table_{spec.key}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = find_sources(spec, Path(a.gguf_dir))
    text = Path(a.text) if a.text else write_wikitext2_test(Path(a.gguf_dir) / "wikitext2_test.txt")
    work = Path(a.work_dir) if a.work_dir else Path("work") / f"m4_{spec.key}"
    llama = Path(a.llama_dir).expanduser()
    configs = standard_configs(a.build_kai, a.build_stock)
    chosen = [configs[c] for c in a.configs]

    sens_path = Path(a.sensitivity) if a.sensitivity else Path(f"results/sensitivity_{spec.key}_kld.json")
    table = SensitivityTable.from_json(sens_path) if sens_path.exists() else None
    order = priority_order(spec, alphabet, table, a.seed)

    work.mkdir(parents=True, exist_ok=True)
    precisions = {p for g in order for p in g} | {spec.fixed_precision}
    need_gb = 1.15 * assembly_space_gb(sources, precisions) + 0.5
    free_gb = shutil.disk_usage(work).free / 1e9
    if free_gb < need_gb:
        raise SystemExit(f"work dir {work.resolve()} has {free_gb:.1f} GB free but assembled genome files need up to {need_gb:.1f} GB "
                         f"(/tmp is RAM-backed on recent Ubuntu): pass --work-dir on a larger disk")
    meter = get_meter(a.meter)
    settings = MeasureSettings(threads=a.threads, ctx=a.ctx, chunks=a.chunks, n_prompt=a.n_prompt, n_gen=a.n_gen,
                               repeats=a.repeats, energy_method=a.energy_method, cooldown_wait_s=a.cooldown_wait)
    meta_path = out.with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path.write_text(json.dumps({PLATFORM_KEY: collect(llama_cpp_dir=llama, extra={"energy_backend": meter.name}),
                                         "args": vars(a), "alphabet": [p.value for p in alphabet],
                                         "configs": {c.label: {"build": c.build, "ppl_args": c.ppl_args, "bench_args": c.bench_args, "env": c.env} for c in chosen}},
                                        indent=2, default=str))
    print(f"{spec.key}: {len(order)} genomes in priority order, configs {a.configs}, meter {meter.name}, threads {a.threads}", flush=True)
    m = GenomeMeasurer(spec, sources, llama, chosen, text, work, meter, settings, idle_s=a.idle_seconds)
    print(f"idle power {m.idle_w:.2f} W; base model perplexity {m.ensure_base():.4f}", flush=True)

    done = done_keys(load_rows(out))
    t_start, n_done, since_idle = time.time(), 0, 0
    for i, g in enumerate(order):
        from evol_inference.tabulated import genome_code
        code = genome_code(g)
        todo = [c.label for c in chosen if (code, c.label) not in done]
        if not todo:
            continue
        if a.limit is not None and n_done >= a.limit:
            print("limit reached"); break
        if a.stop_file and Path(a.stop_file).exists():
            print("stop file found"); break
        if a.budget_seconds is not None and time.time() - t_start > a.budget_seconds:
            print("time budget used"); break
        if since_idle >= a.idle_refresh_every:
            print(f"idle power refreshed: {m.refresh_idle():.2f} W", flush=True)
            since_idle = 0
        for label in (todo if n_done % 2 == 0 else todo[::-1]):  # alternate order between genomes
            try:
                row = m.measure(g, label)
            except Exception as e:  # keep the long run alive; the failure is recorded and the genome retried on resume
                row = {"genome": code, "config": label, "error": f"{type(e).__name__}: {e}"[:300], "t_start": time.time()}
            append_row(out, row)
            if "error" in row:
                print(f"  {code} {label}: ERROR {row['error']}", flush=True)
            else:
                print(f"  {code} {label:9s} KLD {row['kld']:.5f}  decode {row['decode_tps']:7.2f} tok/s  prefill {row['prefill_tps']:8.1f}  "
                      f"{row['joules_per_token']:.4f} J/tok  ({row['wall_s']:.0f}s)", flush=True)
        n_done += 1
        since_idle += 1
        el = time.time() - t_start
        print(f"[{n_done} genomes, {el / 60:.1f} min, {el / n_done:.0f} s/genome]", flush=True)
    print(f"table: {out} ({len(load_rows(out))} rows)")


if __name__ == "__main__":
    main()
