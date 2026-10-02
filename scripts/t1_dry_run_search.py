"""T1 step 7: three-objective search with REAL accuracy and SIMULATED Arm speed/energy.

Accuracy is measured (llama.cpp KL divergence on the 3080 via GgufAssembler + LlamaKldProbe);
decode speed and energy come from `SimulatedArmProbe`, driven by the real per-block parameter
counts and bytes. The result is therefore stamped synthetic and must not be reported as an Arm
measurement -- its purpose is to validate the whole search pipeline end to end before any Mac
time is spent (Doc/implementation_plan_mac-m4.md §2.2).

    PYTHONPATH=. .venv/bin/python scripts/t1_dry_run_search.py --model phi3-mini \\
        --capacity 12 --max-evaluations 40 --stagnation-limit 30          # short check
"""

import argparse
import json
import time
from pathlib import Path

from evol_inference.eval_data import write_wikitext2_test
from evol_inference.genome import Precision
from evol_inference.gguf_assembler import AssembledGgufProvider, GgufAssembler
from evol_inference.llama_probes import LlamaKldProbe
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.mo_ga import MOSteadyStateGA, ParetoRanker, ScalarRanker, run_mo_search, save_mo_snapshot
from evol_inference.model_spec import get_spec
from evol_inference.objectives import Weights
from evol_inference.platform_info import PLATFORM_KEY, collect
from evol_inference.probes import SimulatedArmProbe
from evol_inference.sensitivity import SensitivityTable

LLAMA = Path("~/work/llama.cpp").expanduser()
GGUF = Path("models/gguf")
REAL_BITS = {Precision.FP16: 16.0, Precision.INT8: 8.5, Precision.INT4: 4.5}  # llama.cpp block formats


def sensitivity_seeds(spec, table: SensitivityTable) -> list:
    """Greedy genomes from a sensitivity table: the k most sensitive blocks at the highest
    precision, the rest at the lowest, k = 0..8 (a cheap sweep of the accuracy/size trade-off)."""
    levels = sorted(spec.alphabet, key=lambda p: -p.bits)
    hi, lo = levels[0], levels[-1]
    order = table.ranking(lo)  # most sensitive to the lowest precision first
    return [[hi if i in order[:k] else lo for i in range(8)] for k in range(9)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--ranker", choices=["pareto", "scalar"], default="pareto")
    ap.add_argument("--weights", type=float, nargs=3, default=[1 / 3] * 3, metavar=("W_SPEED", "W_ENERGY", "W_ACCURACY"))
    ap.add_argument("--capacity", type=int, default=40)
    ap.add_argument("--max-evaluations", type=int, default=600)
    ap.add_argument("--stagnation-limit", type=int, default=100)
    ap.add_argument("--mutation-rate", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--build", default="build-cuda")
    ap.add_argument("--seed-sensitivity", action="store_true", help="also seed with greedy sensitivity-guided genomes")
    ap.add_argument("--work-dir", default="/dev/shm")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    spec = get_spec(a.model)
    names = {Precision.FP16: "f16", Precision.INT8: "q8_0-pure", Precision.INT4: "q4_0-pure"}
    sources = {p: GGUF / f"{spec.key}-{n}.gguf" for p, n in names.items() if (GGUF / f"{spec.key}-{n}.gguf").exists()}
    work = Path(a.work_dir)
    asm = GgufAssembler(spec, sources)
    prov = AssembledGgufProvider(asm, work / f"work_{spec.key}.gguf")
    ngl = 99 if "cuda" in a.build else 0
    base = work / f"base_{spec.key}_{spec.reference_precision.value}_c2048_n{a.chunks}_{a.build}.kld"
    acc = LlamaKldProbe(LLAMA / a.build / "bin/llama-perplexity", write_wikitext2_test(GGUF / "wikitext2_test.txt"),
                        prov, base, sources[spec.reference_precision], chunks=a.chunks, n_gpu_layers=ngl)

    params = asm.super_block_param_counts()
    head_bytes = next(int(t.n_bytes) for n, t in asm._tensors[spec.fixed_precision].items() if n == "output.weight")
    arm = SimulatedArmProbe(params, bits=REAL_BITS, fixed_bytes=head_bytes, seed=a.seed)
    ev = MultiObjectiveEvaluator(acc, arm, spec.reference_genome(), params)
    ranker = ParetoRanker() if a.ranker == "pareto" else ScalarRanker(Weights(*a.weights))
    ga = MOSteadyStateGA(ev, capacity=a.capacity, ranker=ranker, seed=a.seed,
                         mutation_rate=a.mutation_rate, alphabet=spec.alphabet)

    seeds = list(spec.baselines().values())
    if a.seed_sensitivity:
        tbl = Path(a.out_dir) / f"sensitivity_{spec.key}_kld.json"
        seeds += sensitivity_seeds(spec, SensitivityTable.from_json(tbl))
    ga.seed(seeds)

    t0 = time.perf_counter()

    def progress(log, ind):
        if log.evaluations % 10 == 0:
            print(f"  eval {log.evaluations:4d}  front={len(ga.population.front()):2d}  "
                  f"HV={log.hypervolume_history[-1]:.4f}  {time.perf_counter() - t0:6.0f}s  "
                  f"assemblies={prov.n_assemblies}", flush=True)

    tag = f"{spec.key}_{a.ranker}_s{a.seed}{a.tag}"
    snap = Path(a.out_dir) / f"dryrun_{tag}_snapshot.json"
    log = run_mo_search(ga, a.max_evaluations, a.stagnation_limit,
                        on_evaluation=lambda l, i: (progress(l, i), l.evaluations % 25 == 0 and save_mo_snapshot(ga, snap)))
    elapsed = time.perf_counter() - t0
    save_mo_snapshot(ga, snap)

    sym = {Precision.FP16: "F", Precision.INT8: "8", Precision.INT4: "4"}
    print(f"\n[SYNTHETIC speed/energy, REAL accuracy] model={spec.key} ranker={a.ranker} alphabet="
          f"{[p.value for p in spec.alphabet]}\n  evals={log.evaluations} stop={log.stopped_reason} "
          f"HV={ga.population.hypervolume():.4f} wall={elapsed:.0f}s ({elapsed / max(1, log.evaluations):.1f}s/eval, "
          f"{prov.n_assemblies} assemblies)  baseline ppl_ref={ev.baseline.perplexity:.4f}")
    print("  baselines (dAcc% / speed gain / energy gain):")
    for name, g in spec.baselines().items():
        v = ev.evaluate(g)
        print(f"    {name:14s} {''.join(sym[p] for p in g)}  {100 * v.accuracy_penalty:7.3f}  {v.speed_gain:+.3f}  {v.energy_gain:+.3f}")
    print("  Pareto front of the final population:")
    for m in sorted(ga.population.front(), key=lambda m: m.vector.accuracy_penalty):
        v = m.vector
        print(f"    {''.join(sym[p] for p in m.genome)}  {100 * v.accuracy_penalty:7.3f}  {v.speed_gain:+.3f}  {v.energy_gain:+.3f}")
    summary = {
        PLATFORM_KEY: collect(extra={"llama_cpp_build": a.build, "work_dir": a.work_dir}),
        "synthetic": True, "synthetic_note": "speed/energy simulated; accuracy measured (KLD, llama.cpp)",
        "model": spec.key, "ranker": a.ranker, "seed": a.seed, "evaluations": log.evaluations,
        "stopped": log.stopped_reason, "wall_s": round(elapsed, 1), "hypervolume": ga.population.hypervolume(),
        "hypervolume_history": log.hypervolume_history,
        "front": [{"genome": [p.value for p in m.genome], "acc_penalty": m.vector.accuracy_penalty,
                   "speed_gain": m.vector.speed_gain, "energy_gain": m.vector.energy_gain} for m in ga.population.front()],
    }
    (Path(a.out_dir) / f"dryrun_{tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
