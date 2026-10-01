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

    def penalty(self, genome: Genome) -> float:
        total = 0.0
        for a, p in zip(self.sensitivity, genome):
            total += {Precision.FP16: 0.0, Precision.INT8: a, Precision.INT4: a * self.int4_factor}[p]
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
                 seed: int = 0):
        if len(param_counts) != N_SUPER_BLOCKS:
            raise ValueError(f"need {N_SUPER_BLOCKS} param counts")
        self.params = list(param_counts)
        self.bw, self.gflops, self.power = bandwidth_gbs * 1e9, prefill_gflops * 1e9, base_power_w
        self.noise = noise
        self.rng = random.Random(seed)

    def _noisy(self, x: float) -> float:
        return x * (1.0 + self.rng.gauss(0.0, self.noise))

    def measure(self, genome: Genome) -> SpeedEnergy:
        bytes_read = sum(n * p.bits / 8 for n, p in zip(self.params, genome))
        t_tok = sum(n * p.bits / 8 / self.bw * self.DEQUANT_DECODE[p]
                    for n, p in zip(self.params, genome))
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
