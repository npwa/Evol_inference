"""Steady-state GA for the multi-objective search (Doc/implementation_plan_mac-m4.md §4.2, D2).

Same asynchronous steady-state structure as `ga.py` (fixed-capacity population, parents
drawn by linear-rank selection, crossover + mutation, evict the worst on overflow) with
the *ordering* factored out into a `Ranker`:

* `ScalarRanker`  -- order by `Weights.fitness` (the README's form; the cross-check).
* `ParetoRanker`  -- order by non-dominated rank then crowding distance (NSGA-II style;
  the headline). Ranks are population-relative, so they are recomputed from the raw
  objective vectors on every insert and never stored.

Crossover/mutation/selection weights are reused from `ga.py` unchanged.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from evol_inference.fitness import genome_key
from evol_inference.ga import crossover, linear_rank_weights, mutate, random_genome
from evol_inference.genome import Genome, Precision
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.objectives import DEFAULT_HV_REF, ObjectiveVector, Weights
from evol_inference.pareto_nd import hypervolume, non_dominated_sort, pareto_order


class Ranker(Protocol):
    def order(self, vectors: Sequence[ObjectiveVector]) -> list[int]:
        """Indices into `vectors`, best first."""


@dataclass(frozen=True)
class ScalarRanker:
    weights: Weights = Weights()

    def order(self, vectors):
        scores = [self.weights.fitness(v) for v in vectors]
        return sorted(range(len(vectors)), key=lambda i: (-scores[i], i))


@dataclass(frozen=True)
class ParetoRanker:
    use_noise_margin: bool = True  # dominance margin = the larger spread of the pair

    def order(self, vectors):
        pts = [v.minimized() for v in vectors]
        eps = max((v.noise_margin for v in vectors), default=0.0) if self.use_noise_margin else 0.0
        return pareto_order(pts, eps)


@dataclass(frozen=True)
class MOIndividual:
    genome: Genome
    vector: ObjectiveVector


class MOPopulation:
    def __init__(self, capacity: int = 40, ranker: Ranker | None = None):
        self.capacity = capacity
        self.ranker = ranker or ParetoRanker()
        self._members: list[MOIndividual] = []

    def __len__(self) -> int:
        return len(self._members)

    def ranked(self) -> list[MOIndividual]:
        """Best first under the current ranker."""
        return [self._members[i] for i in self.ranker.order([m.vector for m in self._members])]

    def insert(self, individual: MOIndividual) -> MOIndividual | None:
        """Add; if over capacity, evict the worst-ranked member and return it. Re-inserting
        a genome already present is a no-op (a duplicate adds no information and would
        crowd out diversity)."""
        key = genome_key(individual.genome)
        if any(genome_key(m.genome) == key for m in self._members):
            return None
        self._members.append(individual)
        if len(self._members) <= self.capacity:
            return None
        order = self.ranker.order([m.vector for m in self._members])
        return self._members.pop(order[-1])

    def front(self) -> list[MOIndividual]:
        pts = [m.vector.minimized() for m in self._members]
        if not pts:
            return []
        return [self._members[i] for i in non_dominated_sort(pts)[0]]

    def hypervolume(self, ref=DEFAULT_HV_REF) -> float:
        return hypervolume([m.vector.minimized() for m in self.front()], ref)

    def refresh_from_cache(self, cache: dict[tuple[str, ...], ObjectiveVector]) -> None:
        """Re-read members' vectors after the evaluator updated them (e.g. `remeasure`)."""
        self._members = [MOIndividual(m.genome, cache[genome_key(m.genome)]) for m in self._members]


class MOSteadyStateGA:
    def __init__(self, evaluator: MultiObjectiveEvaluator, capacity: int = 40,
                 ranker: Ranker | None = None, selection_pressure: float = 1.5,
                 mutation_rate: float = 0.1, crossover_points: int = 1, seed: int | None = None):
        self.evaluator = evaluator
        self.population = MOPopulation(capacity, ranker)
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_points = crossover_points
        self.rng = random.Random(seed)

    def _select_parent(self) -> MOIndividual:
        best_first = self.population.ranked()
        weights = linear_rank_weights(len(best_first), self.selection_pressure)  # 0=worst..n-1=best
        return self.rng.choices(best_first[::-1], weights=weights, k=1)[0]

    def add(self, genome: Genome) -> MOIndividual:
        ind = MOIndividual(genome, self.evaluator.evaluate(genome))
        self.population.insert(ind)
        return ind

    def seed(self, genomes: Sequence[Genome]) -> None:
        """Insert specific genomes (baselines, sensitivity-guided seeds) before random fill."""
        for g in genomes:
            self.add(list(g))

    def step(self) -> MOIndividual:
        a, b = self._select_parent(), self._select_parent()
        child = mutate(crossover(a.genome, b.genome, self.rng, self.crossover_points),
                       self.mutation_rate, self.rng)
        return self.add(child)


@dataclass
class MOSearchLog:
    evaluations: int = 0
    stopped_reason: str = ""
    hypervolume_history: list[float] = field(default_factory=list)


def run_mo_search(ga: MOSteadyStateGA, max_evaluations: int = 600, stagnation_limit: int = 100,
                  hv_ref=DEFAULT_HV_REF, hv_tolerance: float = 1e-9,
                  on_evaluation: Callable[[MOSearchLog, MOIndividual], None] | None = None) -> MOSearchLog:
    """Seed to capacity, then step until the evaluation budget runs out or the population
    front's hypervolume has not improved for `stagnation_limit` real steps (the Pareto
    analogue of 'best fitness did not improve'; seeding does not count toward stagnation,
    for the same reason as in `search.run_search`). Evaluations are counted per call, so a
    cache hit still consumes budget exactly as in the scalar search."""
    log = MOSearchLog()
    best_hv, stale = float("-inf"), 0

    def record(ind: MOIndividual, tracking: bool) -> bool:
        nonlocal best_hv, stale
        log.evaluations += 1
        hv = ga.population.hypervolume(hv_ref)
        log.hypervolume_history.append(hv)
        if on_evaluation:
            on_evaluation(log, ind)
        if hv > best_hv + hv_tolerance:
            best_hv, stale = hv, 0
        elif tracking:
            stale += 1
        if log.evaluations >= max_evaluations:
            log.stopped_reason = "evaluation_budget"
            return True
        if tracking and stale >= stagnation_limit:
            log.stopped_reason = "stagnation"
            return True
        return False

    while len(ga.population) < ga.population.capacity:
        if record(ga.add(random_genome(ga.rng)), tracking=False):
            return log
    while True:
        if record(ga.step(), tracking=True):
            return log


def save_mo_snapshot(ga: MOSteadyStateGA, path: str | Path) -> None:
    """Population + raw-vector cache. Records `synthetic` so a snapshot can never be
    mistaken for measured data."""
    data = {
        "synthetic": ga.evaluator.synthetic,
        "baseline": asdict(ga.evaluator.baseline),
        "population": [{"genome": [p.value for p in m.genome]} for m in ga.population.ranked()],
        "cache": {",".join(k): asdict(v) for k, v in ga.evaluator._cache.items()},
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_mo_snapshot(ga: MOSteadyStateGA, path: str | Path) -> None:
    data = json.loads(Path(path).read_text())
    if data.get("synthetic") and not ga.evaluator.synthetic:
        raise ValueError("snapshot contains synthetic data; refusing to load into a measured run")
    for k, v in data["cache"].items():
        ga.evaluator._cache[tuple(k.split(","))] = ObjectiveVector(**v)
    for entry in data["population"]:
        g = [Precision(x) for x in entry["genome"]]
        ga.population.insert(MOIndividual(g, ga.evaluator._cache[genome_key(g)]))
