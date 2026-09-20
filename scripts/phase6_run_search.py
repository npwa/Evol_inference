"""Phase 6: run the steady-state search (Doc/requirements.md §6, Doc/implementation_plan.md
Phase 6) and report the result against the Phase 5 baselines.

Defaults match the spec (250-300 evaluation budget, 30-evaluation stagnation limit).
Override via CLI flags for a longer, more thorough search of the same 6,561-genome
(8-super-block) space -- e.g. `--max-evaluations 5000 --stagnation-limit 500` for a
multi-hour run that comes much closer to exhausting the space, rather than the ~2%
a spec-default run covers.

Latency is report-only and never feeds fitness (§5/§6); by default this script skips
measuring it during the search itself (saves ~5x the forward passes per evaluation) and
backfills a real measurement only for the final population + baselines afterward. Pass
--measure-latency-during-search for §6's more literal (and much slower) reading.

Snapshots periodically to results/phase6_snapshot.json (crash safety for an unattended
run, §6 step 4) and writes the final population + fitness cache to results/phase6_final.json.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.baselines import BASELINES
from evol_inference.fitness import FitnessEvaluator, backfill_latency
from evol_inference.ga import SteadyStateGA, save_snapshot
from evol_inference.search import DEFAULT_MAX_EVALUATIONS, DEFAULT_STAGNATION_LIMIT, run_search
from evol_inference.weight_bank import WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capacity", type=int, default=40)
    p.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    p.add_argument("--stagnation-limit", type=int, default=DEFAULT_STAGNATION_LIMIT)
    p.add_argument("--mutation-rate", type=float, default=0.1)
    p.add_argument("--selection-pressure", type=float, default=1.5)
    p.add_argument("--crossover-points", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument(
        "--measure-latency-during-search",
        action="store_true",
        help="Measure real latency on every evaluation instead of only backfilling it "
        "for the final report set. Much slower; off by default.",
    )
    p.add_argument("--snapshot-path", default="./results/phase6_snapshot.json")
    p.add_argument(
        "--final-path", default="./results/phase6_final.json",
        help="Give each seed/config a distinct path (e.g. phase6_final_seed1.json) "
        "when running multiple searches to compare, so they don't overwrite each other.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    model.eval()

    bank = WeightBank(model, device="cuda")
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=args.measure_latency_during_search)

    print("Baselines (§7):")
    for name, genome in BASELINES.items():
        r = evaluator.evaluate(genome)
        print(f"  {name:<14} fitness={r.fitness:.4f}  acc_penalty={r.accuracy_penalty:.4f}  eff_gain={r.efficiency_gain:.4f}")
    best_baseline_fitness = max(evaluator.evaluate(g).fitness for g in BASELINES.values())

    ga = SteadyStateGA(
        evaluator,
        capacity=args.capacity,
        selection_pressure=args.selection_pressure,
        mutation_rate=args.mutation_rate,
        crossover_points=args.crossover_points,
        seed=args.seed,
    )

    start = time.perf_counter()

    def on_evaluation(log, individual):
        if log.evaluations % args.snapshot_every == 0:
            save_snapshot(ga, args.snapshot_path)
            elapsed = time.perf_counter() - start
            rate = log.evaluations / elapsed
            print(
                f"  [{log.evaluations:>5} evals, {elapsed:>7.1f}s, {rate:.2f} eval/s] "
                f"best fitness so far = {ga.population.best.result.fitness:.4f}"
            )

    print(
        f"\nRunning steady-state search (capacity={args.capacity}, "
        f"max_evaluations={args.max_evaluations}, stagnation_limit={args.stagnation_limit}, "
        f"measure_latency_during_search={args.measure_latency_during_search})..."
    )
    log = run_search(
        ga, max_evaluations=args.max_evaluations, stagnation_limit=args.stagnation_limit,
        on_evaluation=on_evaluation,
    )

    if not args.measure_latency_during_search:
        print("\nBackfilling real latency for the final population + baselines...")
        backfill_latency(evaluator, [ind.genome for ind in ga.population.ranked()])
        backfill_latency(evaluator, list(BASELINES.values()))
        ga.population.refresh_from_cache(evaluator._cache)

    save_snapshot(ga, args.final_path)

    elapsed = time.perf_counter() - start
    best = ga.population.best
    print(f"\nStopped: {log.stopped_reason} after {log.evaluations} evaluations ({elapsed:.1f}s, {log.evaluations/elapsed:.2f} eval/s)")
    print(f"Best genome found: {[p.value for p in best.genome]}")
    print(
        f"  fitness={best.result.fitness:.4f}  accuracy_penalty={best.result.accuracy_penalty:.4f}  "
        f"efficiency_gain={best.result.efficiency_gain:.4f}  perplexity={best.result.perplexity:.4f}  "
        f"GB={best.result.bytes_ / (1024**3):.3f}  ms/fwd={best.result.latency_ms:.2f}"
    )
    delta = best.result.fitness - best_baseline_fitness
    verdict = "BEAT" if delta > 0 else "DID NOT beat"
    print(f"\nGA {verdict} the best baseline fitness ({best_baseline_fitness:.4f}) by {delta:+.4f}")
    print(f"\nFinal snapshot: {args.final_path}")


if __name__ == "__main__":
    main()
