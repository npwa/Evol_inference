"""Objective vector for the three-way accuracy / speed / power search
(Doc/implementation_plan_mac-m4.md §4.1).

Every normalized term is a fraction of a **fixed** reference measured once on the same
machine (the all-reference-precision genome), never a population-relative rank -- the same
rule as the base project, for the same reason (a cache is only sound if a genome's value
does not depend on who else is in the population).

    accuracy_penalty = (ppl - ppl0) / ppl0            minimize
    speed_gain       = decode_tps / decode_tps0 - 1   maximize   (D1: decode is the search target)
    energy_gain      = 1 - jpt / jpt0                 maximize   (jpt = net joules per token)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    perplexity: float
    decode_tps: float
    joules_per_token: float
    prefill_tps: float = float("nan")


@dataclass(frozen=True)
class SpeedEnergy:
    """One speed + energy measurement of a genome (from a single llama-bench run under
    the energy meter, so both describe the same thermal state)."""
    decode_tps: float
    prefill_tps: float
    joules_per_token: float  # net of idle power
    avg_power_w: float = float("nan")
    decode_tps_spread: float = 0.0  # relative spread (MAD/median) over repeats
    energy_spread: float = 0.0
    synthetic: bool = False


@dataclass(frozen=True)
class ObjectiveVector:
    accuracy_penalty: float
    speed_gain: float
    energy_gain: float
    # raw values behind the normalized terms (reported, and re-derivable)
    perplexity: float
    decode_tps: float
    prefill_tps: float
    joules_per_token: float
    bytes_: float = float("nan")
    # measurement bookkeeping
    n_measurements: int = 1
    speed_spread: float = 0.0
    energy_spread: float = 0.0
    synthetic: bool = False

    def minimized(self) -> tuple[float, float, float]:
        """Objectives as a minimization tuple for `pareto_nd`."""
        return (self.accuracy_penalty, -self.speed_gain, -self.energy_gain)

    @property
    def noise_margin(self) -> float:
        """Largest relative measurement spread of the noisy objectives."""
        return max(self.speed_spread, self.energy_spread)


def make_vector(
    perplexity: float, m: SpeedEnergy, base: Baseline, bytes_: float = float("nan"),
    n_measurements: int = 1,
) -> ObjectiveVector:
    return ObjectiveVector(
        accuracy_penalty=(perplexity - base.perplexity) / base.perplexity,
        speed_gain=m.decode_tps / base.decode_tps - 1.0,
        energy_gain=1.0 - m.joules_per_token / base.joules_per_token,
        perplexity=perplexity,
        decode_tps=m.decode_tps,
        prefill_tps=m.prefill_tps,
        joules_per_token=m.joules_per_token,
        bytes_=bytes_,
        n_measurements=n_measurements,
        speed_spread=m.decode_tps_spread,
        energy_spread=m.energy_spread,
        synthetic=m.synthetic,
    )


@dataclass(frozen=True)
class Weights:
    """Scalarization weights (D2: the cross-check form). Fitness = w_speed*speed_gain +
    w_energy*energy_gain - w_accuracy*accuracy_penalty."""
    w_speed: float = 1 / 3
    w_energy: float = 1 / 3
    w_accuracy: float = 1 / 3

    def fitness(self, v: ObjectiveVector) -> float:
        return (
            self.w_speed * v.speed_gain
            + self.w_energy * v.energy_gain
            - self.w_accuracy * v.accuracy_penalty
        )


# Default hypervolume reference point in `minimized()` space. speed_gain > -1 always (decode
# speed is positive), so 1.0 bounds -speed_gain; 1.0 on accuracy_penalty is a 100% perplexity
# increase; 1.0 on -energy_gain is twice the reference energy per token. A genome worse than
# the reference point on any objective contributes nothing to hypervolume.
DEFAULT_HV_REF = (1.0, 1.0, 1.0)
