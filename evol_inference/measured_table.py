"""A table of measured genomes (JSONL rows from `scripts/m4_measure_table.py`) exposed as probes, so the existing
evaluator, GA, Pareto code and baselines can be run **offline on real measurements**: the Mac day measures, the
analysis (`scripts/m4_analyze.py`) happens on the desktop at no cost.

Repeated measurements of the same (genome, configuration) are combined by the median. Rows are never mixed across
configurations: a table is one configuration. `synthetic` is true if any row came from a mock/replay meter."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Sequence

from evol_inference.genome import Genome
from evol_inference.mac_measure import load_rows
from evol_inference.objectives import SpeedEnergy
from evol_inference.tabulated import genome_code


class MeasuredTable:
    def __init__(self, rows: Sequence[dict], config: str):
        by: dict[str, list[dict]] = {}
        for r in rows:
            if r.get("config") == config and "error" not in r:
                by.setdefault(r["genome"], []).append(r)
        if not by:
            raise ValueError(f"no usable rows for configuration {config!r}")
        self.config, self._rows = config, by
        self.synthetic = any(r.get("synthetic") for rs in by.values() for r in rs)
        ppl0s = {round(r["base_ppl"], 6) for rs in by.values() for r in rs if r.get("base_ppl")}
        if len(ppl0s) > 1:
            raise ValueError(f"rows were measured against different base models (base_ppl {sorted(ppl0s)})")
        self.ppl0 = next(iter(ppl0s)) if ppl0s else float("nan")

    @classmethod
    def from_jsonl(cls, path: str | Path, config: str) -> "MeasuredTable":
        return cls(load_rows(path), config)

    @classmethod
    def configs_in(cls, path: str | Path) -> list[str]:
        return sorted({r["config"] for r in load_rows(path) if "error" not in r})

    def codes(self) -> list[str]:
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def has(self, genome: Genome) -> bool:
        return genome_code(genome) in self._rows

    def _med(self, code: str, key: str) -> float:
        return statistics.median(r[key] for r in self._rows[code])

    def kld(self, genome: Genome) -> float:
        return self._med(self._code(genome), "kld")

    def _code(self, genome: Genome) -> str:
        c = genome_code(genome)
        if c not in self._rows:
            raise KeyError(f"genome {c} was not measured under configuration {self.config!r} ({len(self)} measured)")
        return c

    def repeats(self, genome: Genome) -> int:
        return len(self._rows[self._code(genome)])

    # --- probe interfaces (AccuracyProbe / SpeedEnergyProbe) ---
    def perplexity(self, genome: Genome) -> float:
        return self.ppl0 * math.exp(self.kld(genome))

    def measure(self, genome: Genome) -> SpeedEnergy:
        c = self._code(genome)
        return SpeedEnergy(
            decode_tps=self._med(c, "decode_tps"), prefill_tps=self._med(c, "prefill_tps"),
            joules_per_token=self._med(c, "joules_per_token"), avg_power_w=self._med(c, "avg_power_w"),
            decode_tps_spread=self._med(c, "decode_spread"), energy_spread=0.0,
            synthetic=any(r.get("synthetic") for r in self._rows[c]),
        )
