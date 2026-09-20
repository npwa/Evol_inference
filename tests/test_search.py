"""Unit tests for the Phase 6 search-runner (evol_inference/search.py). The stopping
criterion is pure logic and doesn't need the model -- tested with fake evaluators that
make each stopping path deterministic. Only test_run_search_integration needs the real
GPU-backed evaluator."""

from evol_inference.fitness import FitnessEvaluator, FitnessResult
from evol_inference.ga import SteadyStateGA
from evol_inference.search import run_search


class _IncreasingFitnessEvaluator:
    """Fitness strictly increases every call -- guarantees "improvement" every single
    evaluation, so stagnation never triggers and the evaluation budget is what stops
    the search."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, genome):
        self.calls += 1
        f = self.calls * 0.001
        return FitnessResult(
            fitness=f, accuracy_penalty=0.0, efficiency_gain=f, perplexity=1.0, bytes_=1.0, latency_ms=1.0
        )


class _ConstantFitnessEvaluator:
    """Every genome gets the same fitness -- after the first evaluation (a trivial
    "improvement" from -inf), nothing ever improves again, so stagnation is what stops
    the search."""

    def evaluate(self, genome):
        return FitnessResult(
            fitness=0.5, accuracy_penalty=0.0, efficiency_gain=0.5, perplexity=1.0, bytes_=1.0, latency_ms=1.0
        )


def test_run_search_stops_at_evaluation_budget():
    ga = SteadyStateGA(evaluator=_IncreasingFitnessEvaluator(), capacity=5, mutation_rate=0.5, seed=0)
    log = run_search(ga, max_evaluations=20, stagnation_limit=1000)
    assert log.evaluations == 20
    assert log.stopped_reason == "evaluation_budget"


def test_run_search_stops_at_stagnation():
    ga = SteadyStateGA(evaluator=_ConstantFitnessEvaluator(), capacity=5, mutation_rate=0.5, seed=0)
    log = run_search(ga, max_evaluations=1000, stagnation_limit=3)
    assert log.stopped_reason == "stagnation"
    assert log.evaluations < 1000


def test_stagnation_is_not_counted_during_seeding():
    # Capacity (10) exceeds stagnation_limit (3) -- mirrors the real Phase 6 config
    # (capacity=40 > stagnation_limit=30). With a constant-fitness evaluator, every
    # seed evaluation after the first is a "non-improvement" by definition; if those
    # counted toward stagnation, the search would stop mid-seeding, well before the
    # population even finished filling, and the GA's actual steady-state stepping
    # (selection/crossover/mutation) would never run at all.
    ga = SteadyStateGA(evaluator=_ConstantFitnessEvaluator(), capacity=10, mutation_rate=0.5, seed=0)
    log = run_search(ga, max_evaluations=1000, stagnation_limit=3)
    assert len(ga.population) == 10  # seeding ran to completion
    assert log.stopped_reason == "stagnation"
    # 10 seed evaluations (none counted toward stagnation) + 3 more steady-state steps
    # before stagnation_limit is reached.
    assert log.evaluations == 13


def test_run_search_best_fitness_history_is_monotonically_non_decreasing():
    ga = SteadyStateGA(evaluator=_IncreasingFitnessEvaluator(), capacity=5, mutation_rate=0.5, seed=0)
    log = run_search(ga, max_evaluations=15, stagnation_limit=1000)
    history = log.best_fitness_history
    assert all(b >= a for a, b in zip(history, history[1:]))


def test_run_search_calls_on_evaluation_callback_for_every_evaluation():
    calls = []
    ga = SteadyStateGA(evaluator=_IncreasingFitnessEvaluator(), capacity=5, mutation_rate=0.5, seed=0)
    run_search(
        ga, max_evaluations=8, stagnation_limit=1000, on_evaluation=lambda log, ind: calls.append(ind)
    )
    assert len(calls) == 8


def test_run_search_integration(bank, tokenizer):
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=False)
    ga = SteadyStateGA(evaluator, capacity=4, mutation_rate=0.3, seed=0)
    log = run_search(ga, max_evaluations=6, stagnation_limit=1000)
    assert log.evaluations == 6
    assert log.stopped_reason == "evaluation_budget"
    assert len(ga.population) == 4
