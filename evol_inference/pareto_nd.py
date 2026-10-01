"""N-dimensional Pareto machinery for the multi-objective search (Doc/implementation_plan_mac-m4.md
§4.2): dominance, non-dominated sorting, crowding distance, exact hypervolume.

Everything here is **minimization** on plain float tuples -- callers negate "bigger is
better" objectives first (see `ObjectiveVector.minimized`). Pure math, no model or
hardware dependency. The 2-D plotting helper in `pareto.py` is left as-is.
"""

from __future__ import annotations

import math
from typing import Sequence

Point = Sequence[float]


def dominates(a: Point, b: Point, eps: float = 0.0) -> bool:
    """True if `a` is no worse than `b` on every objective and better by more than `eps`
    on at least one. `eps > 0` is a noise margin: two measured points that differ by less
    than the measurement spread do not dominate each other, so noise cannot evict a
    genuinely non-dominated configuration."""
    if len(a) != len(b):
        raise ValueError("points have different dimensions")
    return all(x <= y for x, y in zip(a, b)) and any(x < y - eps for x, y in zip(a, b))


def non_dominated_sort(points: Sequence[Point], eps: float = 0.0) -> list[list[int]]:
    """Fronts of indices: fronts[0] is the non-dominated set, fronts[1] is non-dominated
    once fronts[0] is removed, and so on (Deb et al., NSGA-II). O(M N^2)."""
    n = len(points)
    dominated_by_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(points[i], points[j], eps):
                dominates_list[i].append(j)
                dominated_by_count[j] += 1
            elif dominates(points[j], points[i], eps):
                dominates_list[j].append(i)
                dominated_by_count[i] += 1
    fronts: list[list[int]] = []
    current = [i for i in range(n) if dominated_by_count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominates_list[i]:
                dominated_by_count[j] -= 1
                if dominated_by_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(points: Sequence[Point], front: Sequence[int]) -> dict[int, float]:
    """NSGA-II crowding distance for the members of one front (index -> distance).
    Boundary points of each objective get +inf so extremes are never crowded out."""
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    n_obj = len(points[front[0]])
    for m in range(n_obj):
        ordered = sorted(front, key=lambda i: points[i][m])
        lo, hi = points[ordered[0]][m], points[ordered[-1]][m]
        dist[ordered[0]] = dist[ordered[-1]] = math.inf
        if hi == lo:
            continue
        for k in range(1, len(ordered) - 1):
            dist[ordered[k]] += (points[ordered[k + 1]][m] - points[ordered[k - 1]][m]) / (hi - lo)
    return dist


def pareto_order(points: Sequence[Point], eps: float = 0.0) -> list[int]:
    """All indices, best first: by front rank, ties broken by larger crowding distance
    (then by index, so the order is deterministic)."""
    order: list[int] = []
    for front in non_dominated_sort(points, eps):
        cd = crowding_distance(points, front)
        order.extend(sorted(front, key=lambda i: (-cd[i], i)))
    return order


def hypervolume(points: Sequence[Point], ref: Point) -> float:
    """Exact hypervolume dominated by `points` and bounded by `ref` (minimization).
    Points not strictly better than `ref` on every objective are clipped away. Recursive
    slicing over the last objective: fine for the <=~100 front points and 3 objectives
    used here (exponential in dimension in general)."""
    pts = [tuple(p) for p in points if all(x < r for x, r in zip(p, ref))]
    return _hv(pts, tuple(ref))


def _hv(pts: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    if not pts:
        return 0.0
    if len(ref) == 1:
        return ref[0] - min(p[0] for p in pts)
    pts = sorted(pts, key=lambda p: p[-1])
    total = 0.0
    for k, p in enumerate(pts):
        upper = pts[k + 1][-1] if k + 1 < len(pts) else ref[-1]
        height = upper - p[-1]
        if height <= 0:
            continue
        # slab between p[-1] and `upper`: every point with last coord <= p[-1] is active
        active = [q[:-1] for q in pts[: k + 1]]
        total += height * _hv(_nondominated(active), ref[:-1])
    return total


def _nondominated(pts: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    return [p for i, p in enumerate(pts) if not any(j != i and dominates(q, p) for j, q in enumerate(pts))]
