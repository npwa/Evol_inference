import math

import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.tabulated import (
    TabulatedAccuracyProbe, enumerate_genomes, front_quality, genome_code, parse_code, true_front,
)

P = Precision


def test_code_roundtrip():
    g = [P.FP16, P.INT8, P.INT4, P.INT8, P.INT4, P.INT4, P.FP16, P.INT8]
    assert genome_code(g) == "F848 44F8".replace(" ", "")
    assert parse_code(genome_code(g)) == g


def test_enumerate_counts_and_order():
    all3 = list(enumerate_genomes(list(P)))
    assert len(all3) == 3 ** N_SUPER_BLOCKS and len({genome_code(g) for g in all3}) == len(all3)
    two = list(enumerate_genomes((P.INT8, P.INT4)))
    assert len(two) == 256 and genome_code(two[0]) == "8" * 8 and genome_code(two[1]) == "8" * 7 + "4"
    assert P.FP16 not in {p for g in two for p in g}


def test_tabulated_probe_matches_kld_formula_and_rejects_unknown(tmp_path):
    g = [P.INT8] * 8
    probe = TabulatedAccuracyProbe({genome_code(g): 0.01}, ppl0=4.5, meta={"build": "x"})
    assert probe.perplexity(g) == pytest.approx(4.5 * math.exp(0.01))
    with pytest.raises(KeyError, match="not in the accuracy table"):
        probe.perplexity([P.INT4] * 8)
    probe.to_json(tmp_path / "t.json")
    again = TabulatedAccuracyProbe.from_json(tmp_path / "t.json")
    assert again.perplexity(g) == probe.perplexity(g) and again.meta == {"build": "x"}


def test_true_front_and_quality():
    pts = {"a": (0.0, 4.0), "b": (1.0, 2.0), "c": (3.0, 1.0), "d": (3.0, 3.0)}  # d dominated
    tf = true_front(pts)
    assert set(tf) == {"a", "b", "c"}
    ref = (5.0, 5.0)
    full = front_quality(list(tf.values()), list(tf.values()), ref)
    assert full["hv_ratio"] == pytest.approx(1.0) and full["recall"] == 1.0
    part = front_quality([pts["a"], pts["c"]], list(tf.values()), ref)
    assert part["hv_ratio"] < 1.0 and part["true_points_found"] == 2 and part["recall"] == pytest.approx(2 / 3)
    # a found point that is dominated (not on the true front) cannot raise the ratio above 1
    assert front_quality([pts["a"], pts["b"], pts["c"], pts["d"]], list(tf.values()), ref)["hv_ratio"] == pytest.approx(1.0)
