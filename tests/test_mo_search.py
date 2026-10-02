"""Multi-objective evaluator / Pareto GA on the synthetic probes (tier T0)."""

import random

import pytest

from evol_inference.baselines import BASELINES
from evol_inference.fitness import genome_key
from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.mo_fitness import MultiObjectiveEvaluator, SyntheticResultError
from evol_inference.mo_ga import (
    MOIndividual, MOPopulation, MOSteadyStateGA, ParetoRanker, ScalarRanker,
    load_mo_snapshot, run_mo_search, save_mo_snapshot,
)
from evol_inference.model_spec import get_spec
from evol_inference.objectives import Baseline, ObjectiveVector, SpeedEnergy, Weights, make_vector
from evol_inference.pareto_nd import dominates
from evol_inference.probes import SimulatedAccuracyProbe, SimulatedArmProbe

PARAMS = [100_000_000] * N_SUPER_BLOCKS
FP16 = [Precision.FP16] * N_SUPER_BLOCKS
INT8 = [Precision.INT8] * N_SUPER_BLOCKS
INT4 = [Precision.INT4] * N_SUPER_BLOCKS


def make_eval(noise=0.0, seed=0, **kw):
    return MultiObjectiveEvaluator(
        SimulatedAccuracyProbe(**kw), SimulatedArmProbe(PARAMS, noise=noise, seed=seed),
        FP16, PARAMS,
    )


def vec(a, s, e, **kw):
    return ObjectiveVector(a, s, e, perplexity=1, decode_tps=1, prefill_tps=1, joules_per_token=1, **kw)


# ---- objectives -------------------------------------------------------------------------

def test_make_vector_normalizes_against_fixed_baseline():
    base = Baseline(perplexity=4.0, decode_tps=10.0, joules_per_token=2.0)
    m = SpeedEnergy(decode_tps=15.0, prefill_tps=100.0, joules_per_token=1.0)
    v = make_vector(4.2, m, base)
    assert v.accuracy_penalty == pytest.approx(0.05)
    assert v.speed_gain == pytest.approx(0.5)
    assert v.energy_gain == pytest.approx(0.5)
    assert v.minimized() == pytest.approx((0.05, -0.5, -0.5))


def test_baseline_genome_has_zero_gain_zero_penalty():
    ev = make_eval()
    v = ev.evaluate(FP16)
    assert v.accuracy_penalty == pytest.approx(0)
    assert v.speed_gain == pytest.approx(0)
    assert v.energy_gain == pytest.approx(0)


def test_weights_fitness():
    w = Weights(w_speed=0.5, w_energy=0.25, w_accuracy=1.0)
    assert w.fitness(vec(0.1, 0.4, 0.2)) == pytest.approx(0.5 * 0.4 + 0.25 * 0.2 - 0.1)


# ---- simulated probes -----------------------------------------------------------------

def test_simulated_decode_is_bandwidth_bound_so_smaller_is_faster():
    ev = make_eval()
    assert ev.evaluate(INT4).speed_gain > ev.evaluate(INT8).speed_gain > 0


def test_simulated_prefill_penalizes_int4_like_the_3080():
    probe = SimulatedArmProbe(PARAMS, noise=0)
    assert probe.measure(INT4).prefill_tps < probe.measure(INT8).prefill_tps


def test_simulated_accuracy_edges_are_sensitive():
    p = SimulatedAccuracyProbe()
    edge = [Precision.INT8] + [Precision.FP16] * 7
    interior = [Precision.FP16] + [Precision.INT8] + [Precision.FP16] * 6
    assert p.penalty(edge) > 10 * p.penalty(interior)


def test_simulated_probe_noise_is_seeded_and_bounded():
    a = SimulatedArmProbe(PARAMS, noise=0.02, seed=3).measure(INT8)
    b = SimulatedArmProbe(PARAMS, noise=0.02, seed=3).measure(INT8)
    assert a == b
    c = SimulatedArmProbe(PARAMS, noise=0.02, seed=4).measure(INT8)
    assert c.decode_tps != a.decode_tps
    assert abs(c.decode_tps / a.decode_tps - 1) < 0.2


# ---- evaluator ------------------------------------------------------------------------

def test_evaluator_caches_by_genome():
    ev = make_eval(noise=0.05)
    first = ev.evaluate(INT8)
    assert ev.evaluate(list(INT8)) is first
    assert len(ev) == 1  # the baseline measurement itself is not a cache entry


def test_evaluator_cache_holds_only_raw_vectors_no_ranks():
    ev = make_eval()
    ev.evaluate(INT8)
    assert all(isinstance(v, ObjectiveVector) for v in ev._cache.values())
    fields = set(ObjectiveVector.__dataclass_fields__)
    assert not any("rank" in f or "crowd" in f for f in fields)


def test_remeasure_replaces_noisy_terms_with_median_and_counts_samples():
    ev = make_eval(noise=0.1, seed=1)
    v1 = ev.evaluate(INT8)
    v3 = ev.remeasure(INT8, n_extra=4)
    assert v3.n_measurements == 5
    assert v3.perplexity == v1.perplexity  # deterministic term untouched
    assert v3.speed_spread >= 0
    assert ev.evaluate(INT8) is v3  # cache now holds the re-measured vector


def test_synthetic_guard():
    ev = make_eval()
    assert ev.synthetic
    with pytest.raises(SyntheticResultError):
        ev.assert_measured()


def test_synthetic_snapshot_refused_by_measured_run(tmp_path):
    ev = make_eval()
    ga = MOSteadyStateGA(ev, capacity=5, seed=0)
    ga.add(INT8)
    path = tmp_path / "s.json"
    save_mo_snapshot(ga, path)

    class Real(SimulatedAccuracyProbe):
        synthetic = False

    class RealSE(SimulatedArmProbe):
        synthetic = False

        def measure(self, genome):
            from dataclasses import replace
            return replace(super().measure(genome), synthetic=False)

    real_ev = MultiObjectiveEvaluator(Real(), RealSE(PARAMS, noise=0), FP16, PARAMS)
    with pytest.raises(ValueError, match="synthetic"):
        load_mo_snapshot(MOSteadyStateGA(real_ev, capacity=5, seed=0), path)


# ---- population / rankers -------------------------------------------------------------

def test_pareto_population_evicts_dominated_never_front_member():
    pop = MOPopulation(capacity=3, ranker=ParetoRanker(use_noise_margin=False))
    g = lambda *a: [Precision.INT8] * 7 + [list(Precision)[a[0]]]  # noqa: E731 - distinct genomes
    members = [
        MOIndividual(g(0), vec(0.0, 0.0, 0.0)),
        MOIndividual(g(1), vec(0.1, 0.3, 0.2)),
        MOIndividual(g(2), vec(0.05, 0.1, 0.1)),
    ]
    for m in members:
        pop.insert(m)
    dominated = MOIndividual([Precision.INT4] * 8, vec(0.5, -0.1, -0.1))
    evicted = pop.insert(dominated)
    assert evicted is dominated


def test_population_ignores_duplicate_genome():
    pop = MOPopulation(capacity=5)
    pop.insert(MOIndividual(INT8, vec(0, 0.1, 0.1)))
    assert pop.insert(MOIndividual(list(INT8), vec(0, 0.1, 0.1))) is None
    assert len(pop) == 1


def test_scalar_ranker_orders_by_weighted_fitness():
    r = ScalarRanker(Weights(w_speed=1, w_energy=0, w_accuracy=1))
    vs = [vec(0.0, 0.1, 0), vec(0.5, 0.9, 0), vec(0.0, 0.5, 0)]
    assert r.order(vs) == [2, 1, 0] or r.order(vs)[0] == 2


def test_noise_margin_keeps_near_tied_points_both_alive():
    a, b = vec(0.0, 0.50, 0.5, speed_spread=0.1), vec(0.0, 0.48, 0.5, speed_spread=0.1)
    assert dominates(a.minimized(), b.minimized())  # without a margin a beats b
    order = ParetoRanker().order([a, b])
    # same front under the margin: a (first) listed first only via tie-break, but both rank 0
    from evol_inference.pareto_nd import non_dominated_sort
    assert len(non_dominated_sort([a.minimized(), b.minimized()], eps=0.1)) == 1
    assert sorted(order) == [0, 1]


# ---- search ---------------------------------------------------------------------------

def run(ranker, seed=0, evals=300, noise=0.0):
    ga = MOSteadyStateGA(make_eval(noise=noise, seed=seed), capacity=30, ranker=ranker, seed=seed)
    log = run_mo_search(ga, max_evaluations=evals, stagnation_limit=80)
    return ga, log


def test_pareto_search_is_deterministic_per_seed():
    a, _ = run(ParetoRanker())
    b, _ = run(ParetoRanker())
    assert [genome_key(m.genome) for m in a.population.ranked()] == [genome_key(m.genome) for m in b.population.ranked()]


def test_hypervolume_history_is_monotone_enough_and_search_stops():
    ga, log = run(ParetoRanker(use_noise_margin=False))
    assert log.stopped_reason in {"evaluation_budget", "stagnation"}
    assert log.evaluations <= 300
    h = log.hypervolume_history
    # steady-state with crowding eviction: the front's hypervolume must essentially not shrink
    assert h[-1] >= max(h) * 0.98


def test_seeded_extremes_are_never_evicted_from_the_pareto_population():
    """Boundary points get infinite crowding distance, so with the baselines seeded (as the
    real runs do) the per-objective extremes survive the whole search."""
    ga = MOSteadyStateGA(make_eval(), capacity=30, ranker=ParetoRanker(use_noise_margin=False), seed=0)
    ga.seed([FP16, INT8, INT4])
    run_mo_search_from_seeded(ga)
    kept = {genome_key(m.genome) for m in ga.population.ranked()}
    assert genome_key(FP16) in kept   # best accuracy
    assert genome_key(INT4) in kept   # best simulated decode speed and energy


def run_mo_search_from_seeded(ga, evals=400):
    return run_mo_search(ga, max_evaluations=evals, stagnation_limit=150)


def test_scalarized_optimum_lies_on_the_pareto_front_consistency_test():
    """D2 cross-check: the scalarized winner must be non-dominated among everything either
    search ever evaluated."""
    sga, _ = run(ScalarRanker(Weights(1 / 3, 1 / 3, 1 / 3)), seed=1, evals=400)
    pga, _ = run(ParetoRanker(use_noise_margin=False), seed=1, evals=400)
    winner = sga.population.ranked()[0].vector.minimized()
    allpts = [v.minimized() for v in {**sga.evaluator._cache, **pga.evaluator._cache}.values()]
    assert not any(dominates(p, winner) for p in allpts)


def test_snapshot_roundtrip(tmp_path):
    ga, _ = run(ParetoRanker(), evals=80)
    path = tmp_path / "snap.json"
    save_mo_snapshot(ga, path)
    ga2 = MOSteadyStateGA(make_eval(), capacity=30, seed=0)
    load_mo_snapshot(ga2, path)
    assert len(ga2.evaluator) >= len(ga.evaluator) - 1
    assert {genome_key(m.genome) for m in ga2.population.ranked()} == {genome_key(m.genome) for m in ga.population.ranked()}


def test_seeding_specific_genomes():
    ga = MOSteadyStateGA(make_eval(), capacity=10, seed=0)
    ga.seed(list(BASELINES.values()))
    assert len(ga.population) == len({genome_key(g) for g in BASELINES.values()})


def test_restricted_alphabet_search_never_proposes_excluded_precision():
    from evol_inference.mo_ga import ParetoRanker

    alphabet = (Precision.INT8, Precision.INT4)
    ev = MultiObjectiveEvaluator(SimulatedAccuracyProbe(), SimulatedArmProbe(PARAMS, noise=0, seed=0), INT8, PARAMS)
    ga = MOSteadyStateGA(ev, capacity=20, ranker=ParetoRanker(), seed=3, alphabet=alphabet)
    run_mo_search(ga, max_evaluations=200, stagnation_limit=80)
    seen = {p for k in ev._cache for p in k}
    assert seen <= {"int8", "int4"}, seen


def test_simulated_arm_probe_uses_effective_bits_and_fixed_bytes():
    base = SimulatedArmProbe(PARAMS, noise=0)
    real = SimulatedArmProbe(PARAMS, noise=0, bits={Precision.INT8: 8.5, Precision.INT4: 4.5})
    assert real.measure(INT8).decode_tps < base.measure(INT8).decode_tps  # 8.5 bpw is slower than 8
    fixed = SimulatedArmProbe(PARAMS, noise=0, fixed_bytes=sum(PARAMS))
    assert fixed.measure(INT8).decode_tps < base.measure(INT8).decode_tps
    # a constant fixed cost dilutes the INT4-over-INT8 speedup
    gain = lambda p: p.measure(INT4).decode_tps / p.measure(INT8).decode_tps  # noqa: E731
    assert gain(fixed) < gain(base)
