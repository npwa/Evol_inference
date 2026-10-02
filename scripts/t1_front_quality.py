"""T1: how good is the GA's Pareto front? Uses an exhaustive accuracy table (real KLD for every
genome) + the NOISELESS simulated Arm model, so the true Pareto front over the whole genome
space is known exactly (Doc/implementation_plan_mac-m4.md §19).

  1. true front + hypervolume over all genomes in the table;
  2. each dry-run result's found front, re-scored noiselessly -> hypervolume ratio, recall;
  3. offline GA studies on the table (seconds, no GPU): rankers, budgets, seeds.

    PYTHONPATH=. .venv/bin/python scripts/t1_front_quality.py --model llama3.1-8b \\
        --runs results/dryrun_llama3.1-8b_pareto_s0.json
"""

import argparse
import json
import statistics
from pathlib import Path

from evol_inference.dryrun_setup import find_sources, sim_arm_probe
from evol_inference.gguf_assembler import GgufAssembler
from evol_inference.genome import Precision
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.mo_ga import MOSteadyStateGA, ParetoRanker, ScalarRanker, run_mo_search
from evol_inference.model_spec import get_spec
from evol_inference.objectives import DEFAULT_HV_REF, Weights
from evol_inference.tabulated import TabulatedAccuracyProbe, front_quality, genome_code, parse_code, true_front

SYM = {Precision.FP16: "F", Precision.INT8: "8", Precision.INT4: "4"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1-8b")
    ap.add_argument("--table", default=None)
    ap.add_argument("--runs", nargs="*", default=[], help="dryrun_*.json files to score against the true front")
    ap.add_argument("--studies", type=int, default=10, help="seeds per offline GA configuration (0 = skip)")
    a = ap.parse_args()

    spec = get_spec(a.model)
    table = TabulatedAccuracyProbe.from_json(a.table or f"results/accuracy_table_{spec.key}.json")
    asm = GgufAssembler(spec, find_sources(spec))
    arm = lambda seed=0, noise=0.0: sim_arm_probe(spec, asm, seed, noise)  # noqa: E731
    ev = MultiObjectiveEvaluator(table, arm(), spec.reference_genome(), asm.super_block_param_counts())

    pts = {c: ev.evaluate(parse_code(c)).minimized() for c in table.kld}
    truth = true_front(pts)
    ref = DEFAULT_HV_REF
    print(f"{spec.key}: {len(pts)} genomes measured, true Pareto front has {len(truth)} points, "
          f"HV={front_quality(list(truth.values()), list(truth.values()), ref)['hv_true']:.4f}")
    print("  true front (genome  dAcc%  speed  energy):")
    for c, p in sorted(truth.items(), key=lambda kv: kv[1][0]):
        print(f"    {c}  {100 * (ev.evaluate(parse_code(c)).accuracy_penalty):7.3f}  {-p[1]:+.3f}  {-p[2]:+.3f}")

    for path in a.runs:
        d = json.loads(Path(path).read_text())
        found = {genome_code([Precision(v) for v in f["genome"]]): None for f in d["front"]}
        found_pts = [ev.evaluate(parse_code(c)).minimized() for c in found]  # re-scored noiselessly
        q = front_quality(found_pts, list(truth.values()), ref)
        print(f"\n{path}: evals={d['evaluations']} stopped={d['stopped']}  found-front size {len(found)}  "
              f"HV ratio {q['hv_ratio']:.4f}  true points found {q['true_points_found']}/{q['true_front_size']} "
              f"(recall {q['recall']:.2f})")

    if a.studies:
        print(f"\nOffline GA studies on the table ({a.studies} seeds each; HV ratio vs true front, mean [min..max]):")
        configs = [
            ("pareto  cap=40 evals=300", dict(capacity=40, ranker=ParetoRanker(), evals=300)),
            ("pareto  cap=40 evals=600", dict(capacity=40, ranker=ParetoRanker(), evals=600)),
            ("pareto  cap=80 evals=600", dict(capacity=80, ranker=ParetoRanker(), evals=600)),
            ("pareto  cap=20 evals=300", dict(capacity=20, ranker=ParetoRanker(), evals=300)),
            ("scalar  cap=40 evals=600 (1/3 each)", dict(capacity=40, ranker=ScalarRanker(Weights()), evals=600)),
        ]
        for name, cfg in configs:
            ratios = []
            for seed in range(a.studies):
                e = MultiObjectiveEvaluator(table, arm(seed, 0.0), spec.reference_genome(), asm.super_block_param_counts())
                ga = MOSteadyStateGA(e, capacity=cfg["capacity"], ranker=cfg["ranker"], seed=seed, alphabet=spec.alphabet)
                ga.seed(list(spec.baselines().values()))
                run_mo_search(ga, cfg["evals"], stagnation_limit=10_000)
                found = [e.evaluate(m.genome).minimized() for m in ga.population.front()]
                ratios.append(front_quality(found, list(truth.values()), ref)["hv_ratio"])
            print(f"  {name:38s} {statistics.mean(ratios):.4f} [{min(ratios):.4f}..{max(ratios):.4f}]")


if __name__ == "__main__":
    main()
