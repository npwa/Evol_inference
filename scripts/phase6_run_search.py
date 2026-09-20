"""Phase 6: run the steady-state search (Doc/requirements.md §6, Doc/implementation_plan.md
Phase 6) and report the result against the Phase 5 baselines.

Snapshots periodically to results/phase6_snapshot.json (crash safety for an unattended
run, §6 step 4) and writes the final population + fitness cache -- which by then also
includes the evaluated baselines -- to results/phase6_final.json.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.baselines import BASELINES
from evol_inference.fitness import FitnessEvaluator
from evol_inference.ga import SteadyStateGA, save_snapshot
from evol_inference.search import run_search
from evol_inference.weight_bank import WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"
POPULATION_CAPACITY = 40
SNAPSHOT_EVERY = 25
SNAPSHOT_PATH = "./results/phase6_snapshot.json"
FINAL_PATH = "./results/phase6_final.json"
SEED = 0


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    model.eval()

    bank = WeightBank(model, device="cuda")
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=True)

    print("Baselines (§7):")
    for name, genome in BASELINES.items():
        r = evaluator.evaluate(genome)
        print(f"  {name:<14} fitness={r.fitness:.4f}  acc_penalty={r.accuracy_penalty:.4f}  eff_gain={r.efficiency_gain:.4f}")
    best_baseline_fitness = max(evaluator.evaluate(g).fitness for g in BASELINES.values())

    ga = SteadyStateGA(
        evaluator,
        capacity=POPULATION_CAPACITY,
        selection_pressure=1.5,
        mutation_rate=0.1,
        crossover_points=1,
        seed=SEED,
    )

    start = time.perf_counter()

    def on_evaluation(log, individual):
        if log.evaluations % SNAPSHOT_EVERY == 0:
            save_snapshot(ga, SNAPSHOT_PATH)
            elapsed = time.perf_counter() - start
            print(
                f"  [{log.evaluations:>4} evals, {elapsed:>6.1f}s] "
                f"best fitness so far = {ga.population.best.result.fitness:.4f}"
            )

    print(f"\nRunning steady-state search (population capacity {POPULATION_CAPACITY})...")
    log = run_search(ga, on_evaluation=on_evaluation)

    save_snapshot(ga, FINAL_PATH)

    elapsed = time.perf_counter() - start
    best = ga.population.best
    print(f"\nStopped: {log.stopped_reason} after {log.evaluations} evaluations ({elapsed:.1f}s)")
    print(f"Best genome found: {[p.value for p in best.genome]}")
    print(
        f"  fitness={best.result.fitness:.4f}  accuracy_penalty={best.result.accuracy_penalty:.4f}  "
        f"efficiency_gain={best.result.efficiency_gain:.4f}  perplexity={best.result.perplexity:.4f}  "
        f"GB={best.result.bytes_ / (1024**3):.3f}  ms/fwd={best.result.latency_ms:.2f}"
    )
    delta = best.result.fitness - best_baseline_fitness
    verdict = "BEAT" if delta > 0 else "DID NOT beat"
    print(f"\nGA {verdict} the best baseline fitness ({best_baseline_fitness:.4f}) by {delta:+.4f}")
    print(f"\nFinal snapshot: {FINAL_PATH}")


if __name__ == "__main__":
    main()
