"""Probe interfaces + synthetic probes for development (Doc/implementation_plan_mac-m4.md §3.2, §2.2.5).

A probe turns a genome into a measurement. Real probes (llama.cpp perplexity,
llama-bench under an `EnergyMeter`) land in later tiers; the synthetic ones here let the
whole three-objective machinery -- evaluator, Pareto GA, reports -- be developed and tested
on any Linux box. **Synthetic results are always stamped `synthetic=True`** and the
evaluator/report code refuses to present them as measurements.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, Sequence

from evol_inference.genome import Genome, N_SUPER_BLOCKS, Precision
from evol_inference.objectives import SpeedEnergy


class AccuracyProbe(Protocol):
    synthetic: bool

    def perplexity(self, genome: Genome) -> float: ...


class SpeedEnergyProbe(Protocol):
    synthetic: bool

    def measure(self, genome: Genome) -> SpeedEnergy: ...


class SimulatedAccuracyProbe:
    """Additive per-block sensitivity model (the README's separable approximation) with a
    small pairwise interaction term: ppl = ppl0 * (1 + sum_i a_i(b_i) + interaction).
    Defaults encode the README's finding: block 0 ~30x more sensitive to INT8 than an
    interior block, block 7 in between, INT4 several times worse than INT8."""
    synthetic = True

    def __init__(self, ppl0: float = 4.37, sensitivity: Sequence[float] | None = None,
                 int4_factor: float = 3.0, interaction: float = 0.0):
        s = list(sensitivity) if sensitivity is not None else [0.011, 0.0004, 0.0004, 0.0004,
                                                               0.0004, 0.0004, 0.0004, 0.002]
        if len(s) != N_SUPER_BLOCKS:
            raise ValueError(f"need {N_SUPER_BLOCKS} sensitivities")
        self.ppl0, self.sensitivity = ppl0, s
        self.int4_factor, self.interaction = int4_factor, interaction

    @classmethod
    def from_table(cls, table, ppl0: float | None = None, interaction: float = 0.0) -> "SimulatedAccuracyProbe":
        """Drive the synthetic probe with *measured* sensitivities (a `SensitivityTable`):
        INT8 penalties from the table, INT4 expressed as a per-block factor over INT8 where
        INT8 is nonzero (falls back to the table's INT4 row directly otherwise)."""
        int8, int4 = table.penalty["int8"], table.penalty["int4"]
        probe = cls(ppl0 or table.reference_perplexity, sensitivity=int8, interaction=interaction)
        probe.int4_row = list(int4)
        return probe

    int4_row = None

    def penalty(self, genome: Genome) -> float:
        total = 0.0
        for i, (a, p) in enumerate(zip(self.sensitivity, genome)):
            int4 = self.int4_row[i] if self.int4_row is not None else a * self.int4_factor
            total += {Precision.FP16: 0.0, Precision.INT8: a, Precision.INT4: int4}[p]
        low = [p is not Precision.FP16 for p in genome]
        total += self.interaction * sum(1 for x, y in zip(low, low[1:]) if x and y)
        return total

    def perplexity(self, genome: Genome) -> float:
        return self.ppl0 * (1.0 + self.penalty(genome))


class SimulatedArmProbe:
    """Analytical decode/prefill/energy model for an Arm CPU -- **synthetic, not a
    measurement**. Decode is bandwidth-bound: tokens/s = bandwidth / bytes_read, plus a
    per-type dequant overhead; prefill is compute-bound with its own per-type factor
    (INT4 dequant makes prefill slower than INT8, mirroring the 3080 finding, so the
    prefill-vs-decode hypotheses of plan D1 can be exercised). Energy = power x time with
    a per-type power factor. Multiplicative noise from a seeded RNG."""
    synthetic = True

    DEQUANT_DECODE = {Precision.FP16: 1.00, Precision.INT8: 1.05, Precision.INT4: 1.15}
    PREFILL_COST = {Precision.FP16: 1.00, Precision.INT8: 0.80, Precision.INT4: 1.10}
    POWER_FACTOR = {Precision.FP16: 1.00, Precision.INT8: 0.95, Precision.INT4: 0.92}

    def __init__(self, param_counts: Sequence[int], bandwidth_gbs: float = 100.0,
                 prefill_gflops: float = 400.0, base_power_w: float = 8.0, noise: float = 0.02,
                 seed: int = 0, bits: dict[Precision, float] | None = None, fixed_bytes: float = 0.0):
        """`bits`: effective bits per weight per precision (default `Precision.bits`; llama.cpp's
        block formats are really 8.5 / 4.5). `fixed_bytes`: bytes read per decoded token outside the
        genome (the output head), a constant that dilutes the genome's speed effect."""
        if len(param_counts) != N_SUPER_BLOCKS:
            raise ValueError(f"need {N_SUPER_BLOCKS} param counts")
        self.bits = {p: float(p.bits) for p in Precision} | (bits or {})
        self.fixed_bytes = fixed_bytes
        self.params = list(param_counts)
        self.bw, self.gflops, self.power = bandwidth_gbs * 1e9, prefill_gflops * 1e9, base_power_w
        self.noise = noise
        self.rng = random.Random(seed)

    @classmethod
    def graviton3(cls, param_counts: Sequence[int], fixed_bytes: float = 0.0, **kw) -> "SimulatedArmProbe":
        """Calibrated to the measured Phi-3-mini run on a c7g.2xlarge (Graviton3, 8 threads, stock
        llama.cpp Arm path; plan §22): read-bandwidth roof 158.7 GB/s; decode reaches 48% / 95% / 72% of it
        for F16 / Q8_0 / Q4_0 (F16 has no optimized kernel; Q4_0 is limited by nibble unpacking), and
        prefill is 10.8 / 93.3 / 108.7 tokens/s. Still a model, not a measurement of another model or CPU,
        and still `synthetic`; energy keeps its placeholder constants (Graviton has no energy counter)."""
        probe = cls(param_counts, bandwidth_gbs=158.7, prefill_gflops=82.0, bits={Precision.INT8: 8.5, Precision.INT4: 4.5},
                    fixed_bytes=fixed_bytes, **kw)
        probe.DEQUANT_DECODE = {Precision.FP16: 1 / 0.48, Precision.INT8: 1 / 0.95, Precision.INT4: 1 / 0.72}
        probe.PREFILL_COST = {Precision.FP16: 1.0, Precision.INT8: 10.8 / 93.3, Precision.INT4: 10.8 / 108.7}
        return probe

    def _noisy(self, x: float) -> float:
        return x * (1.0 + self.rng.gauss(0.0, self.noise))

    def measure(self, genome: Genome) -> SpeedEnergy:
        bytes_read = sum(n * self.bits[p] / 8 for n, p in zip(self.params, genome)) + self.fixed_bytes
        t_tok = sum(n * self.bits[p] / 8 / self.bw * self.DEQUANT_DECODE[p]
                    for n, p in zip(self.params, genome)) + self.fixed_bytes / self.bw
        decode_tps = self._noisy(1.0 / t_tok)
        t_pp = sum(2 * n / self.gflops * self.PREFILL_COST[p] for n, p in zip(self.params, genome))
        prefill_tps = self._noisy(1.0 / t_pp)
        weight = sum(n for n in self.params)
        power = self.power * sum(n * self.POWER_FACTOR[p] for n, p in zip(self.params, genome)) / weight
        jpt = self._noisy(power * t_tok)
        assert bytes_read > 0 and math.isfinite(jpt)
        return SpeedEnergy(
            decode_tps=decode_tps, prefill_tps=prefill_tps, joules_per_token=jpt,
            avg_power_w=power, decode_tps_spread=self.noise, energy_spread=self.noise,
            synthetic=True,
        )
