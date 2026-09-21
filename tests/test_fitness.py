"""Unit tests for the Phase 3 fitness harness (evol_inference/fitness.py). Shared
`tokenizer`/`model`/`bank` fixtures live in conftest.py."""

import torch
import pytest

import math

from evol_inference.fitness import (
    FitnessEvaluator,
    FitnessResult,
    backfill_latency,
    genome_key,
    load_fixed_eval_tokens,
    measure_latency_ms,
)
from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision


@pytest.fixture(scope="session")
def evaluator(bank, tokenizer):
    # measure_latency=False keeps these correctness tests fast; the latency path itself
    # is covered separately by test_measure_latency_returns_positive_value.
    return FitnessEvaluator(bank, tokenizer, measure_latency=False)


def test_load_fixed_eval_tokens_returns_requested_length(tokenizer):
    tokens = load_fixed_eval_tokens(tokenizer, n_tokens=128, device="cuda")
    assert tokens.shape == (1, 128)


def test_measure_latency_returns_positive_value(bank, model):
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    inputs = torch.randint(0, 1000, (1, 64), device=bank.device)
    latency = measure_latency_ms(model, inputs, n_runs=2, n_warmup=1)
    assert latency > 0


def test_fp16_baseline_genome_has_zero_penalty_and_zero_gain(evaluator):
    result = evaluator.evaluate([Precision.FP16] * N_SUPER_BLOCKS)
    assert result.accuracy_penalty == pytest.approx(0.0, abs=1e-9)
    assert result.efficiency_gain == pytest.approx(0.0, abs=1e-9)
    assert result.perplexity == pytest.approx(evaluator.baseline_perplexity)


def test_int8_efficiency_gain_matches_expected_ratio(evaluator):
    result = evaluator.evaluate([Precision.INT8] * N_SUPER_BLOCKS)
    assert result.efficiency_gain == pytest.approx(1 - 8 / 16)


def test_int4_efficiency_gain_matches_expected_ratio(evaluator):
    result = evaluator.evaluate([Precision.INT4] * N_SUPER_BLOCKS)
    assert result.efficiency_gain == pytest.approx(1 - 4 / 16)


def test_int4_genome_degrades_perplexity_relative_to_baseline(evaluator):
    # Not asserting an exact number, just the expected direction for a real
    # quantization scheme: more aggressive quantization should not improve perplexity.
    result = evaluator.evaluate([Precision.INT4] * N_SUPER_BLOCKS)
    assert result.perplexity > evaluator.baseline_perplexity
    assert result.accuracy_penalty > 0


def test_fitness_formula_matches_manual_computation(evaluator):
    result = evaluator.evaluate([Precision.INT8] * N_SUPER_BLOCKS)
    expected = result.efficiency_gain * evaluator.w1 - result.accuracy_penalty * evaluator.w2
    assert result.fitness == pytest.approx(expected)


def test_cache_hit_avoids_recomputation(evaluator, monkeypatch):
    genome = [Precision.INT8, Precision.INT4] * (N_SUPER_BLOCKS // 2)
    first = evaluator.evaluate(genome)

    import evol_inference.fitness as fitness_mod

    calls = []
    original = fitness_mod.compute_perplexity

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fitness_mod, "compute_perplexity", spy)

    second = evaluator.evaluate(genome)  # same genome again -- must be a cache hit
    assert second is first
    assert len(calls) == 0


def test_different_genomes_get_distinct_cache_entries(evaluator):
    evaluator.evaluate([Precision.FP16] * N_SUPER_BLOCKS)
    evaluator.evaluate([Precision.INT8] * N_SUPER_BLOCKS)
    evaluator.evaluate([Precision.INT4] * N_SUPER_BLOCKS)
    assert genome_key([Precision.FP16] * N_SUPER_BLOCKS) in evaluator._cache
    assert genome_key([Precision.INT8] * N_SUPER_BLOCKS) in evaluator._cache
    assert genome_key([Precision.INT4] * N_SUPER_BLOCKS) in evaluator._cache


def test_backfill_latency_replaces_nan_with_a_real_measurement(evaluator):
    genome = [Precision.INT8, Precision.FP16] * (N_SUPER_BLOCKS // 2)
    result = evaluator.evaluate(genome)  # evaluator fixture has measure_latency=False
    assert math.isnan(result.latency_ms)

    backfill_latency(evaluator, [genome])

    updated = evaluator._cache[genome_key(genome)]
    assert updated.latency_ms > 0
    # only latency_ms should change -- everything else stays exactly as computed.
    assert updated.fitness == result.fitness
    assert updated.accuracy_penalty == result.accuracy_penalty
    assert updated.efficiency_gain == result.efficiency_gain
    assert updated.perplexity == result.perplexity
    assert updated.bytes_ == result.bytes_


# --- efficiency_metric="latency" ---------------------------------------------


def test_fitness_result_defaults_to_bytes_metric_for_old_snapshots():
    # Snapshots written before efficiency_metric existed omit the field; it must
    # default rather than fail on FitnessResult(**old_dict).
    r = FitnessResult(
        fitness=0.1, accuracy_penalty=0.0, efficiency_gain=0.1, perplexity=1.0, bytes_=1.0, latency_ms=1.0
    )
    assert r.efficiency_metric == "bytes"


def test_evaluator_rejects_unknown_efficiency_metric(bank, tokenizer):
    with pytest.raises(ValueError):
        FitnessEvaluator(bank, tokenizer, efficiency_metric="flops")


@pytest.fixture(scope="module")
def latency_evaluator(bank, tokenizer):
    # measure_latency=False is passed on purpose, to check the mode overrides it.
    return FitnessEvaluator(bank, tokenizer, efficiency_metric="latency", measure_latency=False)


def test_bytes_mode_is_the_default_and_unchanged(evaluator):
    assert evaluator.efficiency_metric == "bytes"
    result = evaluator.evaluate([Precision.INT4] * N_SUPER_BLOCKS)
    assert result.efficiency_metric == "bytes"
    assert result.efficiency_gain == pytest.approx(1 - 4 / 16)


def test_latency_mode_forces_latency_measurement(latency_evaluator):
    assert latency_evaluator.measure_latency is True
    result = latency_evaluator.evaluate([Precision.INT8] * N_SUPER_BLOCKS)
    assert not math.isnan(result.latency_ms)
    assert result.efficiency_metric == "latency"


def test_latency_mode_efficiency_gain_is_relative_to_fp16_baseline_latency(latency_evaluator):
    result = latency_evaluator.evaluate([Precision.INT8] * N_SUPER_BLOCKS)
    expected_gain = 1.0 - result.latency_ms / latency_evaluator.baseline_latency_ms
    assert result.efficiency_gain == pytest.approx(expected_gain)
    expected_fitness = result.efficiency_gain * latency_evaluator.w1 - result.accuracy_penalty * latency_evaluator.w2
    assert result.fitness == pytest.approx(expected_fitness)


def test_latency_mode_fp16_genome_has_near_zero_efficiency_gain(latency_evaluator):
    # Two separate measurements of the same FP16 config (the baseline at construction,
    # then this evaluation), so exact zero isn't expected -- but observed run-to-run
    # jitter on a 2048-token forward is well under 1%, so 5% is a loose bound.
    result = latency_evaluator.evaluate([Precision.FP16] * N_SUPER_BLOCKS)
    assert abs(result.efficiency_gain) < 0.05
