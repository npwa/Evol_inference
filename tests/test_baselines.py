"""Unit tests for the Phase 5 baseline configurations (evol_inference/baselines.py)."""

import pytest

from evol_inference.baselines import (
    BASELINES,
    fp16_baseline,
    heuristic_baseline,
    int4_baseline,
    int8_baseline,
)
from evol_inference.fitness import FitnessEvaluator
from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision


def test_fp16_baseline_is_all_fp16():
    assert fp16_baseline() == [Precision.FP16] * N_SUPER_BLOCKS


def test_int8_baseline_is_all_int8():
    assert int8_baseline() == [Precision.INT8] * N_SUPER_BLOCKS


def test_int4_baseline_is_all_int4():
    assert int4_baseline() == [Precision.INT4] * N_SUPER_BLOCKS


def test_heuristic_baseline_keeps_outermost_blocks_at_fp16():
    genome = heuristic_baseline()
    assert genome[0] == Precision.FP16
    assert genome[-1] == Precision.FP16


def test_heuristic_baseline_is_most_aggressive_in_the_middle():
    genome = heuristic_baseline()
    middle = genome[len(genome) // 2 - 1 : len(genome) // 2 + 1]
    assert all(p == Precision.INT4 for p in middle)


def test_heuristic_baseline_uses_all_three_precisions():
    assert set(heuristic_baseline()) == {Precision.FP16, Precision.INT8, Precision.INT4}


def test_heuristic_baseline_is_symmetric():
    genome = heuristic_baseline()
    assert genome == genome[::-1]


def test_baselines_dict_has_all_four():
    assert set(BASELINES) == {"fp16", "uniform_int8", "uniform_int4", "heuristic"}


def test_baseline_efficiency_gain_ordering(bank, tokenizer):
    # Guaranteed by construction (bit-widths alone), regardless of model behavior.
    # heuristic's average bit-width (2xFP16 + 4xINT8 + 2xINT4 = 9 bits/block) is
    # *higher* than uniform INT8's (8 bits/block) -- keeping any blocks at full FP16
    # outweighs the INT4 blocks' savings, so heuristic is less efficient than uniform
    # INT8, not in between it and uniform INT4.
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=False)
    results = {name: evaluator.evaluate(genome) for name, genome in BASELINES.items()}

    assert results["fp16"].efficiency_gain == pytest.approx(0.0, abs=1e-9)
    assert results["uniform_int4"].efficiency_gain > results["uniform_int8"].efficiency_gain
    assert results["uniform_int8"].efficiency_gain > results["heuristic"].efficiency_gain
    assert results["heuristic"].efficiency_gain > results["fp16"].efficiency_gain


def test_baseline_accuracy_penalty_ordering(bank, tokenizer):
    # Empirical, not guaranteed by construction -- but overwhelmingly expected for a
    # real quantization scheme: less aggressive quantization should degrade accuracy
    # less. A GA that can't even beat this ordering's intuition would be a red flag.
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=False)
    results = {name: evaluator.evaluate(genome) for name, genome in BASELINES.items()}

    assert results["fp16"].accuracy_penalty == pytest.approx(0.0, abs=1e-9)
    assert results["uniform_int4"].accuracy_penalty >= results["heuristic"].accuracy_penalty
    assert results["heuristic"].accuracy_penalty >= results["uniform_int8"].accuracy_penalty
