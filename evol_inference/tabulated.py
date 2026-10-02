"""Exhaustive accuracy tables (Doc/implementation_plan_mac-m4.md §19).

The genome space is small (3^8 = 6,561 for 3-precision models, 2^8 = 256 for the 8B), and
accuracy is a deterministic function of the genome on a given machine/build. Measuring every
genome once gives (a) the *true* Pareto front, so a search's front can be scored against it
(`front_quality`), and (b) an exact lookup `TabulatedAccuracyProbe` that lets any search be
re-run offline in seconds (ranker / population / mutation studies cost no GPU time).

The table is only valid for the measurement it came from (model, reference, text, ctx, chunks,
llama.cpp build, hardware); `meta` records that, and it is not a substitute for measuring
accuracy on the Arm target.
"""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

from evol_inference.genome import Genome, Precision
from evol_inference.pareto_nd import hypervolume, non_dominated_sort

_CODE = {Precision.FP16: "F", Precision.INT8: "8", Precision.INT4: "4"}
_DECODE = {v: k for k, v in _CODE.items()}


def genome_code(genome: Sequence[Precision]) -> str:
    return "".join(_CODE[p] for p in genome)


def parse_code(code: str) -> Genome:
    return [_DECODE[c] for c in code]


def enumerate_genomes(alphabet: Sequence[Precision], n_blocks: int = 8) -> Iterable[Genome]:
    """Every genome over `alphabet`, lexicographic (last gene varies fastest)."""
    for combo in product(alphabet, repeat=n_blocks):
        yield list(combo)


class TabulatedAccuracyProbe:
    """`genome -> perplexity` from a measured table: `ppl0 * exp(KLD)`, exactly what
    `LlamaKldProbe.perplexity` returns for the same genome."""
    synthetic = False

    def __init__(self, kld_by_code: dict[str, float], ppl0: float, meta: dict | None = None):
        self.kld, self.ppl0, self.meta = dict(kld_by_code), ppl0, dict(meta or {})

    def __len__(self) -> int:
        return len(self.kld)

    def perplexity(self, genome: Genome) -> float:
        try:
            return self.ppl0 * math.exp(self.kld[genome_code(genome)])
        except KeyError:
            raise KeyError(f"genome {genome_code(genome)} is not in the accuracy table "
                           f"({len(self.kld)} entries)") from None

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"ppl0": self.ppl0, "meta": self.meta, "kld": self.kld}))

    @classmethod
    def from_json(cls, path: str | Path) -> "TabulatedAccuracyProbe":
        d = json.loads(Path(path).read_text())
        return cls(d["kld"], d["ppl0"], d.get("meta"))


def true_front(points: dict[str, tuple[float, ...]], eps: float = 0.0) -> dict[str, tuple[float, ...]]:
    """Non-dominated subset of {code: minimized objective tuple}."""
    codes = list(points)
    fronts = non_dominated_sort([points[c] for c in codes], eps)
    return {codes[i]: points[codes[i]] for i in fronts[0]}


def front_quality(found: Sequence[tuple[float, ...]], truth: Sequence[tuple[float, ...]],
                  ref: tuple[float, ...]) -> dict:
    """How good a search's front is against the true front, all in minimization space:
    hypervolume ratio (1.0 = found the whole hypervolume) and the fraction of true-front
    points that were found exactly."""
    hv_f, hv_t = hypervolume(found, ref), hypervolume(truth, ref)
    found_set = {tuple(p) for p in found}
    hit = sum(1 for p in truth if tuple(p) in found_set)
    return {"hv_found": hv_f, "hv_true": hv_t, "hv_ratio": hv_f / hv_t if hv_t else float("nan"),
            "true_front_size": len(truth), "true_points_found": hit,
            "recall": hit / len(truth) if truth else float("nan")}
