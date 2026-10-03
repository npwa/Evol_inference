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
    """Bytes of the output projection read per decoded token. Models with tied embeddings have no
    `output.weight`: the head is then `token_embd.weight` itself."""
    tensors = asm._tensors[spec.fixed_precision]
    return int(tensors["output.weight" if "output.weight" in tensors else "token_embd.weight"].n_bytes)


ARM_MODELS = ("generic", "graviton3")


def sim_arm_probe(spec: ModelSpec, asm: GgufAssembler, seed: int = 0, noise: float = 0.02,
                  model: str = "generic") -> SimulatedArmProbe:
    """`generic`: the original placeholder assumptions (all precisions bandwidth-bound). `graviton3`: calibrated
    to the measured Phi-3 run on a c7g.2xlarge (plan §22): F16 decode only 48% of the memory roof, Q8_0 95%,
    Q4_0 72%, and the measured prefill ratios. Calibrated on Phi-3 and applied to other models as an assumption."""
    if model not in ARM_MODELS:
        raise ValueError(f"arm model must be one of {ARM_MODELS}, got {model!r}")
    params, fixed = asm.super_block_param_counts(), head_bytes(asm, spec)
    if model == "graviton3":
        return SimulatedArmProbe.graviton3(params, fixed_bytes=fixed, noise=noise, seed=seed)
    return SimulatedArmProbe(params, bits=REAL_BITS, noise=noise, seed=seed, fixed_bytes=fixed)
