import math
import random

import pytest

from evol_inference.pareto_nd import (
    crowding_distance, dominates, hypervolume, non_dominated_sort, pareto_order,
)


def test_dominates_basic():
    assert dominates((1, 1), (2, 2))
    assert dominates((1, 2), (2, 2))
    assert not dominates((1, 1), (1, 1))  # equal points do not dominate
    assert not dominates((1, 3), (2, 2))  # trade-off


def test_dominates_noise_margin():
    assert dominates((1.0, 1.0), (1.05, 1.0))
    assert not dominates((1.0, 1.0), (1.05, 1.0), eps=0.1)  # within spread: neither wins
    assert dominates((1.0, 1.0), (1.5, 1.0), eps=0.1)


def test_dominates_dimension_mismatch():
    with pytest.raises(ValueError):
        dominates((1, 2), (1, 2, 3))


def test_non_dominated_sort_fronts():
    pts = [(1, 4), (2, 2), (4, 1), (3, 3), (5, 5)]
    fronts = non_dominated_sort(pts)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert fronts[1] == [3]
    assert fronts[2] == [4]


def test_non_dominated_sort_partitions_all_points_and_front0_is_nondominated():
    rng = random.Random(1)
    pts = [tuple(rng.random() for _ in range(3)) for _ in range(60)]
    fronts = non_dominated_sort(pts)
    assert sorted(i for f in fronts for i in f) == list(range(60))
    for i in fronts[0]:
        assert not any(dominates(pts[j], pts[i]) for j in range(60))
    for later in fronts[1:]:  # everything past front 0 is dominated by something earlier
        for i in later:
            assert any(dominates(pts[j], pts[i]) for j in range(60))


def test_crowding_distance_boundaries_infinite_and_interior_finite():
    pts = [(0, 4), (1, 3), (2, 1), (4, 0)]
    cd = crowding_distance(pts, [0, 1, 2, 3])
    assert cd[0] == cd[3] == math.inf
    assert 0 < cd[1] < math.inf and 0 < cd[2] < math.inf


def test_crowding_distance_prefers_isolated_point():
    pts = [(0, 10), (1, 9), (1.1, 8.9), (5, 5), (10, 0)]
    cd = crowding_distance(pts, list(range(5)))
    assert cd[3] > cd[2]  # (5,5) is far from neighbours, (1.1,8.9) is crowded


def test_pareto_order_ranks_then_crowding():
    pts = [(1, 4), (2, 2), (4, 1), (3, 3), (5, 5)]
    order = pareto_order(pts)
    assert set(order[:3]) == {0, 1, 2} and order[3] == 3 and order[4] == 4


def test_hypervolume_known_cases():
    assert hypervolume([(1, 1)], (3, 3)) == pytest.approx(4.0)
    assert hypervolume([(1, 2), (2, 1)], (3, 3)) == pytest.approx(3.0)  # L-shape: 2+2-1
    assert hypervolume([(1, 1, 1)], (2, 2, 2)) == pytest.approx(1.0)
    assert hypervolume([(0, 0, 0)], (2, 3, 4)) == pytest.approx(24.0)
    # two 3-D points: union of boxes 2x1x1 and 1x2x1 overlapping in 1x1x1 -> 2+2-1
    assert hypervolume([(0, 1, 0), (1, 0, 0)], (2, 2, 1)) == pytest.approx(3.0)


def test_hypervolume_ignores_dominated_and_out_of_ref_points():
    base = hypervolume([(1, 1)], (3, 3))
    assert hypervolume([(1, 1), (2, 2)], (3, 3)) == pytest.approx(base)
    assert hypervolume([(1, 1), (4, 0.5)], (3, 3)) == pytest.approx(base)
    assert hypervolume([], (3, 3)) == 0.0


def test_hypervolume_matches_monte_carlo_3d():
    rng = random.Random(7)
    pts = [tuple(rng.random() for _ in range(3)) for _ in range(12)]
    ref = (1.0, 1.0, 1.0)
    hv = hypervolume(pts, ref)
    n, hit = 40000, 0
    for _ in range(n):
        s = (rng.random(), rng.random(), rng.random())
        if any(all(p[k] <= s[k] for k in range(3)) for p in pts):
            hit += 1
    assert hv == pytest.approx(hit / n, abs=0.01)


def test_hypervolume_monotone_in_added_points():
    pts = [(0.5, 0.5, 0.5)]
    h1 = hypervolume(pts, (1, 1, 1))
    h2 = hypervolume(pts + [(0.2, 0.9, 0.9)], (1, 1, 1))
    assert h2 > h1
