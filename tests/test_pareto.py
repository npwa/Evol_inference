"""Unit tests for Pareto front computation (evol_inference/pareto.py) -- pure math, no
GPU needed."""

from evol_inference.pareto import pareto_front_indices


def test_pure_tradeoff_keeps_all_points():
    # For maximize-x/minimize-y, a genuine tradeoff needs y to increase *with* x (the
    # gain costs you) -- both axes moving together, not opposite, is what makes this a
    # real frontier where no point dominates another.
    x = [1, 2, 3]
    y = [1, 2, 3]
    assert pareto_front_indices(x, y, maximize_x=True, minimize_y=True) == [0, 1, 2]


def test_same_x_better_y_dominates():
    x = [2, 2]
    y = [5, 3]  # same x; index 1's strictly better y dominates index 0
    assert pareto_front_indices(x, y, maximize_x=True, minimize_y=True) == [1]


def test_strictly_worse_on_both_axes_is_excluded():
    x = [2, 1]
    y = [1, 5]  # index 1 (x=1, y=5) is worse than index 0 (x=2, y=1) on both axes
    assert pareto_front_indices(x, y, maximize_x=True, minimize_y=True) == [0]


def test_duplicate_points_both_kept():
    # Neither duplicate strictly dominates the other -- both belong on the front.
    x = [1, 1]
    y = [5, 5]
    assert pareto_front_indices(x, y, maximize_x=True, minimize_y=True) == [0, 1]


def test_minimize_x_direction():
    # bytes (lower=better) vs accuracy_penalty (lower=better) -- both minimized.
    x = [3, 2, 1]
    y = [1, 2, 3]
    result = pareto_front_indices(x, y, maximize_x=False, minimize_y=True)
    assert sorted(result) == [0, 1, 2]  # still a pure tradeoff, all non-dominated


def test_single_best_point_dominates_everything():
    x = [5, 1, 2]
    y = [0.1, 0.9, 0.5]  # index 0: best on both axes
    assert pareto_front_indices(x, y, maximize_x=True, minimize_y=True) == [0]
