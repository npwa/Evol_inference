"""Phase 4: steady-state GA engine (Doc/requirements.md §6, Doc/implementation_plan.md
Phase 4).

One live population (fixed capacity, sorted by fitness) that workers continuously pull
parents from and insert children into -- no generations, no synchronization barrier.
Elitism isn't a special rule here: an individual only leaves the population when a
strictly better one is inserted and the population is at capacity, so the best
individuals already in the population are never evicted by construction.
"""

from __future__ import annotations

import bisect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from evol_inference.fitness import FitnessEvaluator, FitnessResult
from evol_inference.weight_bank import Genome, N_SUPER_BLOCKS, Precision

DEFAULT_CAPACITY = 30  # midpoint of requirements.md §6's 20-50 range


def linear_rank_weights(n: int, s: float = 1.5) -> list[float]:
    """Selection probability per rank (0 = worst .. n-1 = best -- rank increases with
    fitness, the standard Baker linear-ranking convention), per §6:
    `P(rank i) = (2-s)/N + 2*i*(s-1) / (N*(N-1))`, for `1.0 <= s <= 2.0`. `s = 1.0` gives
    exactly uniform random selection; `s = 2.0` gives the sharpest linear falloff, with
    the single worst individual reaching zero selection probability."""
    if n == 1:
        return [1.0]
    if not (1.0 <= s <= 2.0):
        raise ValueError(f"selection pressure s must be in [1.0, 2.0], got {s}")
    return [(2 - s) / n + 2 * i * (s - 1) / (n * (n - 1)) for i in range(n)]


def random_genome(rng: random.Random) -> Genome:
    return [rng.choice(list(Precision)) for _ in range(N_SUPER_BLOCKS)]


def crossover(parent_a: Genome, parent_b: Genome, rng: random.Random, n_points: int = 1) -> Genome:
    """Single- or multi-point crossover on the per-layer gene array (§6)."""
    length = len(parent_a)
    assert len(parent_b) == length
    n_points = max(0, min(n_points, length - 1))
    points = sorted(rng.sample(range(1, length), n_points)) if n_points > 0 else []
    boundaries = [0, *points, length]
    child: Genome = []
    from_a = True
    for start, end in zip(boundaries, boundaries[1:]):
        child.extend((parent_a if from_a else parent_b)[start:end])
        from_a = not from_a
    return child


def mutate(genome: Genome, rate: float, rng: random.Random) -> Genome:
    """Per-gene random reassignment to a *different* precision level (§6), each gene
    independently mutated with probability `rate`."""
    mutated = list(genome)
    for i, gene in enumerate(mutated):
        if rng.random() < rate:
            mutated[i] = rng.choice([p for p in Precision if p != gene])
    return mutated


@dataclass(frozen=True)
class Individual:
    genome: Genome
    result: FitnessResult


class SteadyStatePopulation:
    """A fixed-capacity, fitness-sorted (ascending) population: index 0 is the eviction
    candidate, index -1 is the current best."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = capacity
        self._individuals: list[Individual] = []

    def __len__(self) -> int:
        return len(self._individuals)

    def ranked(self) -> list[Individual]:
        """Best-first order, matching the selection formula's rank convention (rank 0 =
        best)."""
        return list(reversed(self._individuals))

    @property
    def best(self) -> Individual | None:
        return self._individuals[-1] if self._individuals else None

    def insert(self, individual: Individual) -> Individual | None:
        """Insert, keeping ascending-fitness order; evict the worst if this pushes the
        population over capacity. Returns the evicted individual, or None if nothing
        was evicted."""
        bisect.insort(self._individuals, individual, key=lambda ind: ind.result.fitness)
        if len(self._individuals) > self.capacity:
            return self._individuals.pop(0)
        return None


class SteadyStateGA:
    """Ties selection, crossover, mutation, and the fitness evaluator together into the
    steady-state worker step from §6."""

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        capacity: int = DEFAULT_CAPACITY,
        selection_pressure: float = 1.5,
        mutation_rate: float = 0.1,
        crossover_points: int = 1,
        seed: int | None = None,
    ):
        self.evaluator = evaluator
        self.population = SteadyStatePopulation(capacity)
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.crossover_points = crossover_points
        self.rng = random.Random(seed)

    def _select_parent(self) -> Individual:
        # population._individuals is already ascending by fitness (worst..best), which
        # is exactly linear_rank_weights' rank convention (0 = worst, n-1 = best) -- no
        # reordering needed, unlike `ranked()` (best-first, for display/reporting).
        individuals = self.population._individuals
        weights = linear_rank_weights(len(individuals), self.selection_pressure)
        return self.rng.choices(individuals, weights=weights, k=1)[0]

    def seed_random(self, n: int) -> None:
        """Bootstrap the population with `n` random genomes before steady-state
        stepping begins -- selection needs at least one individual to draw parents from,
        and two to do crossover."""
        for _ in range(n):
            genome = random_genome(self.rng)
            result = self.evaluator.evaluate(genome)
            self.population.insert(Individual(genome=genome, result=result))

    def step(self) -> Individual:
        """One worker step (§6): select two parents, crossover + mutate into a child,
        evaluate (cache-aware, §5/§6), insert into the population."""
        parent_a = self._select_parent()
        parent_b = self._select_parent()
        child_genome = crossover(parent_a.genome, parent_b.genome, self.rng, self.crossover_points)
        child_genome = mutate(child_genome, self.mutation_rate, self.rng)
        result = self.evaluator.evaluate(child_genome)
        individual = Individual(genome=child_genome, result=result)
        self.population.insert(individual)
        return individual


def save_snapshot(ga: SteadyStateGA, path: str | Path) -> None:
    """Persist both the population and the full fitness cache (§6 step 4) so a crash
    doesn't lose an unattended run -- scalar metadata only, never model weights or
    tensors."""
    data = {
        "population": [
            {"genome": [p.value for p in ind.genome], "result": asdict(ind.result)}
            for ind in ga.population.ranked()
        ],
        "cache": {
            ",".join(key): asdict(result) for key, result in ga.evaluator._cache.items()
        },
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_snapshot(ga: SteadyStateGA, path: str | Path) -> None:
    """Restore the evaluator's cache and the population from a snapshot written by
    `save_snapshot`, so an interrupted run can resume without re-evaluating anything
    already cached."""
    data = json.loads(Path(path).read_text())
    for key_str, result_dict in data["cache"].items():
        ga.evaluator._cache[tuple(key_str.split(","))] = FitnessResult(**result_dict)
    for entry in data["population"]:
        genome = [Precision(v) for v in entry["genome"]]
        result = FitnessResult(**entry["result"])
        ga.population.insert(Individual(genome=genome, result=result))
