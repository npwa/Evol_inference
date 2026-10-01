"""Multi-objective evaluator (Doc/implementation_plan_mac-m4.md §4). Plays the role
`FitnessEvaluator` plays for the GPU path, for three objectives and noisy measurements.

Cache contract: only **raw per-genome objective vectors** are stored. Anything
population-relative (Pareto rank, crowding distance, hypervolume) is recomputed from the
vectors whenever the population changes -- a cached rank would silently go stale the
moment another individual is inserted (the bug the base project's fixed-baseline
normalization exists to avoid).
"""

from __future__ import annotations

import statistics
from dataclasses import replace

from evol_inference.fitness import genome_key
from evol_inference.genome import Genome
from evol_inference.objectives import Baseline, ObjectiveVector, make_vector
from evol_inference.probes import AccuracyProbe, SpeedEnergyProbe


class SyntheticResultError(RuntimeError):
    """Raised when synthetic (simulated) data is about to be presented as measured."""


class MultiObjectiveEvaluator:
    def __init__(
        self,
        accuracy: AccuracyProbe,
        speed_energy: SpeedEnergyProbe,
        reference_genome: Genome,
        param_counts: list[int] | None = None,
        baseline_repeats: int = 3,
    ):
        self.accuracy = accuracy
        self.speed_energy = speed_energy
        self.param_counts = param_counts
        self._cache: dict[tuple[str, ...], ObjectiveVector] = {}
        self._samples: dict[tuple[str, ...], list] = {}  # raw SpeedEnergy repeats per genome
        self.synthetic = bool(getattr(accuracy, "synthetic", False) or getattr(speed_energy, "synthetic", False))

        # Fixed-reference baseline (median of repeats: it normalizes every other value).
        runs = [speed_energy.measure(reference_genome) for _ in range(baseline_repeats)]
        self.baseline = Baseline(
            perplexity=accuracy.perplexity(reference_genome),
            decode_tps=statistics.median(r.decode_tps for r in runs),
            joules_per_token=statistics.median(r.joules_per_token for r in runs),
            prefill_tps=statistics.median(r.prefill_tps for r in runs),
        )
        self.reference_genome = list(reference_genome)

    def bytes_for(self, genome: Genome) -> float:
        if self.param_counts is None:
            return float("nan")
        return sum(n * p.bits / 8 for n, p in zip(self.param_counts, genome))

    def evaluate(self, genome: Genome) -> ObjectiveVector:
        key = genome_key(genome)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        m = self.speed_energy.measure(genome)
        self._samples[key] = [m]
        vec = make_vector(self.accuracy.perplexity(genome), m, self.baseline, self.bytes_for(genome))
        self._cache[key] = vec
        return vec

    def remeasure(self, genome: Genome, n_extra: int = 2) -> ObjectiveVector:
        """Take `n_extra` more speed/energy measurements (perplexity is deterministic and
        is not repeated) and replace the cached noisy terms with the median over all
        samples. Used for elites before they may be reported (plan §4.3)."""
        key = genome_key(genome)
        vec = self.evaluate(genome)
        samples = self._samples[key]
        samples.extend(self.speed_energy.measure(genome) for _ in range(n_extra))
        med = lambda xs: statistics.median(xs)  # noqa: E731
        decode = [s.decode_tps for s in samples]
        energy = [s.joules_per_token for s in samples]
        rel = lambda xs: (statistics.median(abs(x - med(xs)) for x in xs) / med(xs)) if med(xs) else 0.0  # noqa: E731
        merged = replace(
            samples[0], decode_tps=med(decode), joules_per_token=med(energy),
            prefill_tps=med([s.prefill_tps for s in samples]),
            decode_tps_spread=rel(decode), energy_spread=rel(energy),
        )
        new = make_vector(vec.perplexity, merged, self.baseline, vec.bytes_, n_measurements=len(samples))
        self._cache[key] = new
        return new

    def assert_measured(self) -> None:
        """Call before producing a 'measured results' report."""
        if self.synthetic or any(v.synthetic for v in self._cache.values()):
            raise SyntheticResultError(
                "results include synthetic (simulated) probe data; refusing to report as measured"
            )

    def __len__(self) -> int:
        return len(self._cache)
