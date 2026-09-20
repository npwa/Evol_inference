"""Phase 7: Pareto front computation (Doc/requirements.md §8) -- pure math, no model
dependency, used to highlight the non-dominated configs in the accuracy-vs-efficiency
and accuracy-vs-latency report plots.
"""

from __future__ import annotations


def pareto_front_indices(
    x: list[float], y: list[float], maximize_x: bool = True, minimize_y: bool = True
) -> list[int]:
    """Indices of the non-dominated points in (x, y), sorted by x ascending.

    Point i is dominated by point j if j is at least as good as i on both axes and
    strictly better on at least one -- "better" meaning larger x if `maximize_x`
    (smaller otherwise) and smaller y if `minimize_y` (larger otherwise)."""
    n = len(x)
    assert len(y) == n
    x_sign = 1 if maximize_x else -1
    y_sign = -1 if minimize_y else 1  # sign both axes so "larger is always better"

    def at_least_as_good(a: int, b: int) -> bool:
        return x_sign * x[a] >= x_sign * x[b] and y_sign * y[a] >= y_sign * y[b]

    def strictly_better(a: int, b: int) -> bool:
        return (x_sign * x[a] > x_sign * x[b]) or (y_sign * y[a] > y_sign * y[b])

    front = [
        i
        for i in range(n)
        if not any(j != i and at_least_as_good(j, i) and strictly_better(j, i) for j in range(n))
    ]
    return sorted(front, key=lambda i: x[i])
