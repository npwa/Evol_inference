"""Shared setup for the T1 dry-run / exhaustive / front-quality scripts: locate a model's GGUF
sources and build the simulated Arm probe from the real per-block parameter counts and the
real output-head bytes (so the speed model sees the actual tensor sizes)."""

from __future__ import annotations

from pathlib import Path

from evol_inference.genome import Precision
from evol_inference.gguf_assembler import GgufAssembler
from evol_inference.model_spec import ModelSpec
from evol_inference.probes import SimulatedArmProbe

GGUF_DIR = Path("models/gguf")
# effective bits per weight of llama.cpp's block formats (scales included)
REAL_BITS = {Precision.FP16: 16.0, Precision.INT8: 8.5, Precision.INT4: 4.5}
_FILE = {Precision.FP16: "f16", Precision.INT8: "q8_0-pure", Precision.INT4: "q4_0-pure"}


def find_sources(spec: ModelSpec, gguf_dir: Path = GGUF_DIR) -> dict[Precision, Path]:
    found = {p: gguf_dir / f"{spec.key}-{n}.gguf" for p, n in _FILE.items()}
    return {p: path for p, path in found.items() if path.exists()}


def head_bytes(asm: GgufAssembler, spec: ModelSpec) -> int:
    return next(int(t.n_bytes) for n, t in asm._tensors[spec.fixed_precision].items() if n == "output.weight")


def sim_arm_probe(spec: ModelSpec, asm: GgufAssembler, seed: int = 0, noise: float = 0.02) -> SimulatedArmProbe:
    return SimulatedArmProbe(asm.super_block_param_counts(), bits=REAL_BITS, noise=noise, seed=seed,
                             fixed_bytes=head_bytes(asm, spec))
