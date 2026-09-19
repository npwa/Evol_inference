"""Phase 5: baseline configurations to compare the GA against (Doc/requirements.md §7,
Doc/implementation_plan.md Phase 5). The heuristic baseline is what the GA needs to beat
to make the project's point (§7) -- everything else is a sanity bound (FP16 for
accuracy, uniform INT4 for efficiency).
"""

from __future__ import annotations

from evol_inference.weight_bank import Genome, N_SUPER_BLOCKS, Precision


def fp16_baseline() -> Genome:
    """Upper bound on accuracy, worst on efficiency (§7)."""
    return [Precision.FP16] * N_SUPER_BLOCKS


def int8_baseline() -> Genome:
    """Naive uniform baseline (§7)."""
    return [Precision.INT8] * N_SUPER_BLOCKS


def int4_baseline() -> Genome:
    """Naive uniform baseline (§7) -- best efficiency bound, worst accuracy among the
    baselines."""
    return [Precision.INT4] * N_SUPER_BLOCKS


def heuristic_baseline() -> Genome:
    """"Quantize middle layers more aggressively than first/last layers" (§7) -- a
    common published heuristic, based on layers near the input/output typically being
    more sensitive to quantization than the interior. Graded across the 8 super-blocks:
    the outermost pair stays FP16, the next pair in from each side drops to INT8, and
    the innermost pair goes to INT4."""
    return [
        Precision.FP16,  # 0 (outermost)
        Precision.INT8,  # 1
        Precision.INT8,  # 2
        Precision.INT4,  # 3 (innermost)
        Precision.INT4,  # 4 (innermost)
        Precision.INT8,  # 5
        Precision.INT8,  # 6
        Precision.FP16,  # 7 (outermost)
    ]


BASELINES: dict[str, Genome] = {
    "fp16": fp16_baseline(),
    "uniform_int8": int8_baseline(),
    "uniform_int4": int4_baseline(),
    "heuristic": heuristic_baseline(),
}
