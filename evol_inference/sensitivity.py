"""Per-block quantization sensitivity table (Doc/implementation_plan_mac-m4.md §2.2.2).

For each super-block i and precision b, the accuracy penalty of quantizing *only* block i to
b while every other block stays at the reference precision:

    a_i(b) = (ppl(ref with gene i = b) - ppl(ref)) / ppl(ref)

This is the `a_i(b)` of the README's separability analysis, measured directly: N_blocks x
{precisions} + 1 evaluations (17 for the usual 8 x {INT8, INT4}). Uses: seeding the search
with sensitivity-guided genomes, an additive surrogate to sanity-check real accuracy
(`additivity_residual`), driving the synthetic probe with measured values, and the
cross-model / cross-backend profile comparison pre-registered in plan D5
(`spearman`). Probe-agnostic: takes any `genome -> perplexity` callable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from evol_inference.genome import Genome, N_SUPER_BLOCKS, Precision


@dataclass
class SensitivityTable:
    reference_perplexity: float
    reference_precision: str
    # precision value -> per-block penalty (len N_SUPER_BLOCKS), single-block-quantized
    penalty: dict[str, list[float]]
    # precision value -> measured penalty of the *uniform* genome (all blocks at that precision)
    uniform_penalty: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)  # model, backend, window, platform...

    def additive_prediction(self, precision: Precision) -> float:
        return sum(self.penalty[precision.value])

    def additivity_residual(self, precision: Precision) -> float:
        """measured uniform penalty minus the sum of single-block penalties. Zero if the
        penalty is exactly additive; positive means blocks hurt each other when quantized together."""
        return self.uniform_penalty[precision.value] - self.additive_prediction(precision)

    def share(self, precision: Precision) -> list[float]:
        """Each block's fraction of the summed single-block penalty (sums to 1)."""
        total = self.additive_prediction(precision)
        return [p / total for p in self.penalty[precision.value]]

    def ranking(self, precision: Precision) -> list[int]:
        """Block indices, most sensitive first."""
        pen = self.penalty[precision.value]
        return sorted(range(len(pen)), key=lambda i: -pen[i])

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SensitivityTable":
        return cls(**json.loads(Path(path).read_text()))


def measure_sensitivity(
    perplexity: Callable[[Genome], float],
    reference_precision: Precision = Precision.FP16,
    precisions: Sequence[Precision] = (Precision.INT8, Precision.INT4),
    measure_uniform: bool = True,
    meta: dict | None = None,
    n_blocks: int = N_SUPER_BLOCKS,
) -> SensitivityTable:
    ref = [reference_precision] * n_blocks
    ppl0 = perplexity(ref)
    penalty: dict[str, list[float]] = {}
    uniform: dict[str, float] = {}
    for p in precisions:
        if p is reference_precision:
            raise ValueError("precisions must differ from the reference precision")
        row = []
        for i in range(n_blocks):
            g = list(ref)
            g[i] = p
            row.append((perplexity(g) - ppl0) / ppl0)
        penalty[p.value] = row
        if measure_uniform:
            uniform[p.value] = (perplexity([p] * n_blocks) - ppl0) / ppl0
    return SensitivityTable(ppl0, reference_precision.value, penalty, uniform, dict(meta or {}))


def _ranks(xs: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (Pearson on average ranks). The pre-registered measure of
    whether two sensitivity profiles (models, backends, windows) agree."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need two equal-length sequences of at least 2 values")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        raise ValueError("a constant sequence has no rank correlation")
    return cov / (va * vb) ** 0.5


def edge_dominance(row: Sequence[float]) -> float:
    """Mean penalty of the two edge blocks (first, last) over the mean of the interior blocks:
    how strongly sensitivity sits at the edges (the README's finding 3). >1 means edges are
    more sensitive than the interior."""
    interior = row[1:-1]
    return ((row[0] + row[-1]) / 2) / (sum(interior) / len(interior))
