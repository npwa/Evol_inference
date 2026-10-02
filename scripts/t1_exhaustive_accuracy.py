"""T1: measure accuracy (KLD vs the original model) for EVERY genome of a model's alphabet and
save the table (Doc/implementation_plan_mac-m4.md §19). 8B: 2^8 = 256 genomes (~1 h on the 3080);
3-precision models: 6,561 (hours -- resumable, run in the background).

Resumable: re-running skips genomes already in the table. Writes every --save-every genomes.

    PYTHONPATH=. .venv/bin/python scripts/t1_exhaustive_accuracy.py --model llama3.1-8b
"""

import argparse
import time
from pathlib import Path

from evol_inference.dryrun_setup import find_sources
from evol_inference.eval_data import write_wikitext2_test
from evol_inference.gguf_assembler import AssembledGgufProvider, GgufAssembler
from evol_inference.llama_probes import LlamaKldProbe
from evol_inference.model_spec import get_spec
from evol_inference.platform_info import PLATFORM_KEY, collect
from evol_inference.tabulated import TabulatedAccuracyProbe, enumerate_genomes, genome_code

LLAMA = Path("~/work/llama.cpp").expanduser()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--build", default="build-cuda")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--work-dir", default="/dev/shm")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-every", type=int, default=25)
    a = ap.parse_args()

    spec = get_spec(a.model)
    sources = find_sources(spec)
    work = Path(a.work_dir)
    prov = AssembledGgufProvider(GgufAssembler(spec, sources), work / f"work_{spec.key}.gguf")
    base = work / f"base_{spec.key}_{spec.reference_precision.value}_c2048_n{a.chunks}_{a.build}.kld"
    probe = LlamaKldProbe(LLAMA / a.build / "bin/llama-perplexity", write_wikitext2_test(Path("models/gguf/wikitext2_test.txt")),
                          prov, base, sources[spec.reference_precision], chunks=a.chunks,
                          n_gpu_layers=99 if "cuda" in a.build else 0)
    out = Path(a.out or f"results/accuracy_table_{spec.key}.json")
    meta = {"model": spec.key, "backend": f"llama.cpp {a.build}", "metric": "kld vs original model",
            "alphabet": [p.value for p in spec.alphabet], "chunks": a.chunks, "ctx": 2048,
            "reference": spec.reference_precision.value, PLATFORM_KEY: collect()}

    ppl0 = probe.ensure_base()  # saves the base logits if absent; ppl0 of the original model
    if out.exists():
        table = TabulatedAccuracyProbe.from_json(out)
        if abs(table.ppl0 - ppl0) > 1e-3 * ppl0:
            raise SystemExit(f"{out} was measured with a different base (ppl0 {table.ppl0} vs {ppl0}); refusing to mix")
    else:
        table = TabulatedAccuracyProbe({}, ppl0, meta)
    genomes = list(enumerate_genomes(spec.alphabet))
    todo = [g for g in genomes if genome_code(g) not in table.kld]
    print(f"{spec.key}: {len(genomes)} genomes, {len(table)} already measured, {len(todo)} to do", flush=True)
    t0 = time.perf_counter()
    for i, g in enumerate(todo, 1):
        table.kld[genome_code(g)] = probe.measure(g).mean_kld
        if i % a.save_every == 0 or i == len(todo):
            table.to_json(out)
            rate = (time.perf_counter() - t0) / i
            print(f"  {len(table)}/{len(genomes)}  {rate:.1f}s/genome  eta {rate * (len(todo) - i) / 60:.0f} min", flush=True)
    table.to_json(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
