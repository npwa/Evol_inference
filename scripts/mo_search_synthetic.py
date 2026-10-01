"""Tier-T0 dry run of the three-objective search on SYNTHETIC probes (no model, no GPU, no
Arm hardware): exercises the evaluator, the Pareto/scalar GA, snapshots and reporting end
to end. Every number it prints is simulated -- the output is stamped synthetic and the
report path refuses to treat it as measured (Doc/implementation_plan_mac-m4.md §4.3).

    python scripts/mo_search_synthetic.py --model llama3.2-3b --ranker pareto --seed 0
"""

import argparse

from evol_inference.baselines import BASELINES
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.mo_ga import (
    MOSteadyStateGA, ParetoRanker, ScalarRanker, run_mo_search, save_mo_snapshot,
)
from evol_inference.model_spec import MODEL_SPECS, get_spec
from evol_inference.objectives import Weights
from evol_inference.platform_info import PLATFORM_KEY, collect
from evol_inference.probes import SimulatedAccuracyProbe, SimulatedArmProbe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODEL_SPECS), default="phi3-mini")
    ap.add_argument("--ranker", choices=["pareto", "scalar"], default="pareto")
    ap.add_argument("--weights", type=float, nargs=3, default=[1 / 3] * 3,
                    metavar=("W_SPEED", "W_ENERGY", "W_ACCURACY"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity", type=int, default=40)
    ap.add_argument("--max-evaluations", type=int, default=600)
    ap.add_argument("--stagnation-limit", type=int, default=100)
    ap.add_argument("--out", default="results/synthetic_mo_snapshot.json")
    a = ap.parse_args()

    spec = get_spec(a.model)
    # parameters per super-block follow the model's (possibly uneven) grouping
    params = [s * 100_000_000 for s in spec.sizes]
    ev = MultiObjectiveEvaluator(
        SimulatedAccuracyProbe(), SimulatedArmProbe(params, seed=a.seed),
        spec.reference_genome(), params,
    )
    ranker = ParetoRanker() if a.ranker == "pareto" else ScalarRanker(Weights(*a.weights))
    ga = MOSteadyStateGA(ev, capacity=a.capacity, ranker=ranker, seed=a.seed)
    ga.seed(list(BASELINES.values()))
    log = run_mo_search(ga, a.max_evaluations, a.stagnation_limit)

    print(f"[SYNTHETIC] model={spec.key} sizes={spec.sizes} ranker={a.ranker} "
          f"evals={log.evaluations} stop={log.stopped_reason} HV={ga.population.hypervolume():.4f}")
    for m in sorted(ga.population.front(), key=lambda m: -m.vector.speed_gain)[:12]:
        v = m.vector
        print(f"  {''.join(p.value[-1] if p.value != 'fp16' else 'F' for p in m.genome)}  "
              f"dAcc={v.accuracy_penalty:+.4f} speed={v.speed_gain:+.3f} energy={v.energy_gain:+.3f}")
    save_mo_snapshot(ga, a.out)
    print(f"snapshot -> {a.out} (platform: {collect()['machine']}; {PLATFORM_KEY} block is added by real runs)")


if __name__ == "__main__":
    main()
