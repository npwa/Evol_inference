"""Sensitivity-table logic against the synthetic accuracy probe (tier T0)."""

import json

import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.probes import SimulatedAccuracyProbe
from evol_inference.sensitivity import (
    SensitivityTable, edge_dominance, measure_sensitivity, spearman,
)

SENS = [0.011, 0.0004, 0.0005, 0.0003, 0.0004, 0.0006, 0.0004, 0.002]


def table(interaction=0.0):
    probe = SimulatedAccuracyProbe(sensitivity=SENS, int4_factor=3.0, interaction=interaction)
    return measure_sensitivity(probe.perplexity, meta={"model": "synthetic"}), probe


def test_recovers_the_probes_per_block_penalties_and_counts_evaluations():
    calls = []
    probe = SimulatedAccuracyProbe(sensitivity=SENS)

    def counting(g):
        calls.append(g)
        return probe.perplexity(g)

    t = measure_sensitivity(counting)
    assert len(calls) == 1 + 2 * N_SUPER_BLOCKS + 2  # reference + 2x8 single-block + 2 uniform
    assert t.penalty["int8"] == pytest.approx(SENS)
    assert t.penalty["int4"] == pytest.approx([3 * s for s in SENS])


def test_additive_probe_has_zero_residual_and_interaction_shows_up():
    t, _ = table(interaction=0.0)
    assert t.additivity_residual(Precision.INT8) == pytest.approx(0, abs=1e-12)
    t2, _ = table(interaction=0.001)
    assert t2.additivity_residual(Precision.INT8) == pytest.approx(0.001 * (N_SUPER_BLOCKS - 1))
    assert t2.additivity_residual(Precision.INT8) > 0  # blocks hurt each other when quantized together


def test_share_and_ranking_put_block_zero_first():
    t, _ = table()
    assert sum(t.share(Precision.INT8)) == pytest.approx(1.0)
    assert t.share(Precision.INT8)[0] == pytest.approx(0.011 / sum(SENS))
    assert t.ranking(Precision.INT8)[:2] == [0, 7]


def test_json_roundtrip(tmp_path):
    t, _ = table()
    t.to_json(tmp_path / "s.json")
    assert SensitivityTable.from_json(tmp_path / "s.json") == t
    json.loads((tmp_path / "s.json").read_text())


def test_rejects_reference_in_precisions():
    with pytest.raises(ValueError):
        measure_sensitivity(lambda g: 1.0, Precision.FP16, [Precision.FP16])


def test_non_fp16_reference_for_8b_models():
    probe = lambda g: 5.0 + sum(0.1 for p in g if p is Precision.INT4)  # noqa: E731
    t = measure_sensitivity(probe, Precision.INT8, [Precision.INT4])
    assert t.reference_precision == "int8" and t.penalty["int4"] == pytest.approx([0.02] * 8)


def test_simulated_probe_from_table_reproduces_measured_penalties():
    t, _ = table()
    sim = SimulatedAccuracyProbe.from_table(t)
    for p in (Precision.INT8, Precision.INT4):
        for i in range(N_SUPER_BLOCKS):
            g = [Precision.FP16] * N_SUPER_BLOCKS
            g[i] = p
            assert sim.perplexity(g) / t.reference_perplexity - 1 == pytest.approx(t.penalty[p.value][i])


# ---- statistics -----------------------------------------------------------------------

def test_spearman_known_values():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)


def test_spearman_handles_ties_and_rejects_degenerate_input():
    assert spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        spearman([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError):
        spearman([1], [1])
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


def test_edge_dominance():
    assert edge_dominance([0.011, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.002]) > 10
    assert edge_dominance([1, 1, 1, 1, 1, 1, 1, 1]) == pytest.approx(1.0)
    assert edge_dominance([0, 1, 1, 1, 1, 1, 1, 0]) == pytest.approx(0.0)
