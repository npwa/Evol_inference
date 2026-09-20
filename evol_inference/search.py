"""Phase 6: run the steady-state search until its stopping criterion (Doc/requirements.md
§6, Doc/implementation_plan.md Phase 6).

Stops at whichever comes first: a fixed evaluation budget (250-300 total evaluations --
~4.5% of the 3^8 = 6,561 genome search space at 300), or stagnation (the population's
best fitness hasn't improved for 30 consecutive evaluations). Every evaluation counts
toward the *budget*, including the genomes used to seed the initial population -- from a
compute-budget perspective seeding costs exactly as much as any other evaluation.

Stagnation is different: it's only tracked once real steady-state stepping begins, not
during the initial random-genome seeding phase. Seeding is randomly sampling the search
space to bootstrap a population, so a long run of non-improving *random* draws is
expected and means nothing about whether the GA is stuck -- it isn't evidence of
anything, since selection/crossover/mutation haven't run yet. Whenever the population
capacity is close to or exceeds the stagnation limit (as it will be at the default
capacity=40, stagnation_limit=30), counting seeding evaluations toward stagnation can
exhaust the entire stagnation budget on ordinary sampling noise before the GA gets a
single real step -- confirmed in practice: the first run of this search stopped after
just 46 evaluations, only 6 of which were real steady-state steps.

`population.best` is monotonically non-decreasing over a run by construction (§6
elitism: an individual only leaves the population when something strictly better is
inserted), so "hasn't improved" is exactly "hasn't increased".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from evol_inference.ga import Individual, SteadyStateGA, random_genome

DEFAULT_MAX_EVALUATIONS = 300
DEFAULT_STAGNATION_LIMIT = 30


@dataclass
class SearchLog:
    evaluations: int = 0
    stopped_reason: str = ""
    best_fitness_history: list[float] = field(default_factory=list)


def run_search(
    ga: SteadyStateGA,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
    on_evaluation: Callable[[SearchLog, Individual], None] | None = None,
) -> SearchLog:
    """Seed the population to capacity with random genomes, then run steady-state
    worker steps (§6) until the stopping criterion is met. `on_evaluation`, if given, is
    called after every evaluation (seed or step) with `(log, individual)` -- useful for
    progress reporting or periodic snapshotting."""
    log = SearchLog()
    best_so_far = float("-inf")
    evaluations_since_improvement = 0

    def _record(individual: Individual, track_stagnation: bool) -> bool:
        """Update stopping-criterion state after one evaluation; returns True if the
        search should stop now. `track_stagnation=False` during seeding (see module
        docstring) -- the evaluation budget still counts seeding either way."""
        nonlocal best_so_far, evaluations_since_improvement
        log.evaluations += 1
        current_best = ga.population.best.result.fitness
        log.best_fitness_history.append(current_best)
        if on_evaluation is not None:
            on_evaluation(log, individual)

        if current_best > best_so_far:
            best_so_far = current_best
            evaluations_since_improvement = 0
        elif track_stagnation:
            evaluations_since_improvement += 1

        if log.evaluations >= max_evaluations:
            log.stopped_reason = "evaluation_budget"
            return True
        if track_stagnation and evaluations_since_improvement >= stagnation_limit:
            log.stopped_reason = "stagnation"
            return True
        return False

    while len(ga.population) < ga.population.capacity:
        genome = random_genome(ga.rng)
        result = ga.evaluator.evaluate(genome)
        individual = Individual(genome=genome, result=result)
        ga.population.insert(individual)
        if _record(individual, track_stagnation=False):
            return log

    while True:
        individual = ga.step()
        if _record(individual, track_stagnation=True):
            return log
