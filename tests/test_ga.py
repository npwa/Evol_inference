"""Unit tests for the Phase 4 GA engine (evol_inference/ga.py).

Selection/crossover/mutation/population logic is pure Python and doesn't touch the
model, so most tests here construct a `SteadyStateGA` with `evaluator=None` and insert
individuals directly rather than going through `evaluator.evaluate()`. Only
`test_ga_step_integration` needs the real GPU-backed evaluator (shared `bank`/
`tokenizer` fixtures from conftest.py).
"""

import random

import pytest

from evol_inference.fitness import FitnessEvaluator, FitnessResult
from evol_inference.ga import (
    Individual,
    SteadyStateGA,
    SteadyStatePopulation,
    crossover,
    linear_rank_weights,
    load_snapshot,
    mutate,
    random_genome,
    save_snapshot,
)
from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision


def _fake_result(fitness: float) -> FitnessResult:
    return FitnessResult(
        fitness=fitness,
        accuracy_penalty=0.0,
        efficiency_gain=fitness,
        perplexity=10.0,
        bytes_=1.0,
        latency_ms=1.0,
    )


def _fake_individual(fitness: float, genome=None) -> Individual:
    genome = genome or [Precision.FP16] * N_SUPER_BLOCKS
    return Individual(genome=genome, result=_fake_result(fitness))


# --- linear_rank_weights ---------------------------------------------------


def test_linear_rank_weights_sum_to_one_and_favor_best():
    weights = linear_rank_weights(10, s=1.5)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[-1] > weights[0]  # rank n-1 = best, most likely


def test_linear_rank_weights_uniform_when_s_is_one():
    weights = linear_rank_weights(10, s=1.0)
    assert all(w == pytest.approx(weights[0]) for w in weights)


def test_linear_rank_weights_single_individual():
    assert linear_rank_weights(1, s=1.5) == [1.0]


def test_linear_rank_weights_rejects_out_of_range_s():
    with pytest.raises(ValueError):
        linear_rank_weights(10, s=2.5)
    with pytest.raises(ValueError):
        linear_rank_weights(10, s=1.0 - 1e-9)


# --- crossover / mutate -----------------------------------------------------


def test_crossover_child_is_combination_of_both_parents():
    rng = random.Random(0)
    parent_a = [Precision.FP16] * N_SUPER_BLOCKS
    parent_b = [Precision.INT4] * N_SUPER_BLOCKS
    child = crossover(parent_a, parent_b, rng, n_points=1)
    assert len(child) == N_SUPER_BLOCKS
    assert all(gene in (Precision.FP16, Precision.INT4) for gene in child)
    assert set(child) == {Precision.FP16, Precision.INT4}  # a real split happened


def test_crossover_multi_point_stays_within_bounds():
    rng = random.Random(0)
    parent_a = random_genome(rng)
    parent_b = random_genome(rng)
    child = crossover(parent_a, parent_b, rng, n_points=7)  # >= length - 1
    assert len(child) == N_SUPER_BLOCKS


def test_mutate_always_changes_selected_genes():
    rng = random.Random(0)
    genome = [Precision.FP16] * N_SUPER_BLOCKS
    mutated = mutate(genome, rate=1.0, rng=rng)  # every gene mutates
    assert all(g != Precision.FP16 for g in mutated)


def test_mutate_rate_zero_never_changes_genome():
    rng = random.Random(0)
    genome = random_genome(rng)
    assert mutate(genome, rate=0.0, rng=rng) == genome


# --- SteadyStatePopulation ---------------------------------------------------


def test_population_ranked_order_best_first():
    pop = SteadyStatePopulation(capacity=10)
    for fitness in [0.3, 0.9, 0.1, 0.5]:
        pop.insert(_fake_individual(fitness))
    ranked_fitness = [ind.result.fitness for ind in pop.ranked()]
    assert ranked_fitness == sorted(ranked_fitness, reverse=True)
    assert pop.best.result.fitness == 0.9


def test_population_evicts_worst_when_over_capacity():
    pop = SteadyStatePopulation(capacity=3)
    for fitness in [0.1, 0.2, 0.3]:
        pop.insert(_fake_individual(fitness))
    evicted = pop.insert(_fake_individual(0.9))
    assert evicted.result.fitness == 0.1  # the worst, not the newest
    assert len(pop) == 3
    assert pop.best.result.fitness == 0.9


def test_population_never_evicts_the_best_regardless_of_insertion_order():
    pop = SteadyStatePopulation(capacity=3)
    pop.insert(_fake_individual(0.99))  # best, inserted first
    for fitness in [0.1, 0.2, 0.3, 0.15, 0.25]:
        pop.insert(_fake_individual(fitness))
    assert pop.best.result.fitness == 0.99
    assert len(pop) == 3


def test_refresh_from_cache_updates_results_and_preserves_order():
    pop = SteadyStatePopulation(capacity=3)
    genomes = [random_genome(random.Random(i)) for i in range(3)]
    for genome, fitness in zip(genomes, [0.1, 0.3, 0.2]):
        pop.insert(Individual(genome=genome, result=_fake_result(fitness)))

    ranked_before = [ind.result.fitness for ind in pop.ranked()]

    # Simulate an out-of-band cache update (e.g. fitness.backfill_latency) that only
    # changes latency_ms, keeping fitness the same for every genome.
    from evol_inference.ga import genome_key

    cache = {}
    for ind in pop.ranked():
        key = genome_key(ind.genome)
        cache[key] = FitnessResult(
            fitness=ind.result.fitness,
            accuracy_penalty=ind.result.accuracy_penalty,
            efficiency_gain=ind.result.efficiency_gain,
            perplexity=ind.result.perplexity,
            bytes_=ind.result.bytes_,
            latency_ms=999.0,  # the "backfilled" value
        )

    pop.refresh_from_cache(cache)

    assert [ind.result.fitness for ind in pop.ranked()] == ranked_before  # order preserved
    assert all(ind.result.latency_ms == 999.0 for ind in pop.ranked())


# --- selection bias -----------------------------------------------------


def test_select_parent_favors_better_individuals_statistically():
    ga = SteadyStateGA(evaluator=None, capacity=10, selection_pressure=2.0, seed=0)
    for i, fitness in enumerate([i / 10 for i in range(10)]):
        ga.population.insert(_fake_individual(fitness))

    counts = {}
    for _ in range(2000):
        parent = ga._select_parent()
        counts[parent.result.fitness] = counts.get(parent.result.fitness, 0) + 1

    best_count = counts.get(0.9, 0)
    worst_count = counts.get(0.0, 0)
    assert best_count > worst_count * 5  # s=2.0 is the sharpest pressure available


# --- integration (needs the real GPU-backed evaluator) -----------------------


def test_ga_step_integration(bank, tokenizer):
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=False)
    ga = SteadyStateGA(evaluator, capacity=4, mutation_rate=0.3, seed=0)

    ga.seed_random(4)
    assert len(ga.population) == 4

    ga.step()
    assert len(ga.population) == 4  # at capacity, insert triggers an eviction
    assert all(isinstance(ind.result, FitnessResult) for ind in ga.population.ranked())


# --- snapshot persistence (no GPU needed -- FitnessResult data is plain scalars) ------


class _DummyEvaluator:
    def __init__(self):
        self._cache = {}


def test_snapshot_roundtrip(tmp_path):
    ga = SteadyStateGA(evaluator=_DummyEvaluator(), capacity=5, seed=0)
    genomes = [random_genome(random.Random(i)) for i in range(3)]
    for i, genome in enumerate(genomes):
        result = _fake_result(fitness=i / 10)
        ga.evaluator._cache[tuple(p.value for p in genome)] = result
        ga.population.insert(Individual(genome=genome, result=result))

    path = tmp_path / "snapshot.json"
    save_snapshot(ga, path)

    restored = SteadyStateGA(evaluator=_DummyEvaluator(), capacity=5, seed=0)
    load_snapshot(restored, path)

    assert len(restored.population) == len(ga.population)
    assert len(restored.evaluator._cache) == len(ga.evaluator._cache)
    assert restored.population.best.result.fitness == ga.population.best.result.fitness
