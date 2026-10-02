"""Per-model description: how a model's transformer layers group into the 8 genome genes,
which GGUF tensors belong to each layer, and which precision is the fixed-baseline
reference. The genome is always 8 genes for every model (Doc/implementation_plan_mac-m4.md
D5), so models whose layer count is not divisible by 8 get near-equal uneven groups.

Tensor names are llama.cpp's GGUF names. They are verified against real converted GGUF
files in the llama.cpp-tier tests (`@pytest.mark.llamacpp`); the table below is the single
place to fix them if a conversion disagrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from evol_inference.genome import N_SUPER_BLOCKS, Precision

# Genome precision -> GGML tensor type (as gguf-py's GGMLQuantizationType names).
GGML_TYPE_NAME: dict[Precision, str] = {
    Precision.FP16: "F16",
    Precision.INT8: "Q8_0",
    Precision.INT4: "Q4_0",
}

_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)\.weight$")


def group_sizes(n_layers: int, n_groups: int = N_SUPER_BLOCKS) -> list[int]:
    """Near-equal partition of `n_layers` into `n_groups`, larger groups first
    (28 -> 4,4,4,4,3,3,3,3). Exactly equal when divisible (32 -> eight 4s)."""
    if n_layers < n_groups:
        raise ValueError(f"need at least {n_groups} layers, got {n_layers}")
    base, extra = divmod(n_layers, n_groups)
    return [base + (1 if i < extra else 0) for i in range(n_groups)]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_repo: str
    n_layers: int
    # GGUF tensor-name stems (the `X` in `blk.N.X.weight`) that the genome controls.
    linear_tensors: tuple[str, ...]
    # Precision of the fixed-baseline reference genome. FP16 unless it will not fit.
    reference_precision: Precision = Precision.FP16
    # Precision used for tensors the genome does not control (embeddings, output head).
    fixed_precision: Precision = Precision.INT8
    params_b: float = 0.0
    gated: bool = False  # Hugging Face licence acceptance required
    # Precisions a gene may take. Models whose F16 weights do not fit (8B on a 10 GB GPU /
    # 24 GiB Mac with headroom) search {INT8, INT4} only.
    alphabet: tuple[Precision, ...] = (Precision.FP16, Precision.INT8, Precision.INT4)

    @property
    def sizes(self) -> list[int]:
        return group_sizes(self.n_layers)

    def group_of_layer(self, layer: int) -> int:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.key} ({self.n_layers} layers)")
        end = 0
        for g, size in enumerate(self.sizes):
            end += size
            if layer < end:
                return g
        raise AssertionError("unreachable")

    def layers_of_group(self, group: int) -> range:
        start = sum(self.sizes[:group])
        return range(start, start + self.sizes[group])

    def genome_tensor_names(self, layer: int) -> list[str]:
        return [f"blk.{layer}.{stem}.weight" for stem in self.linear_tensors]

    def classify_tensor(self, name: str) -> int | None:
        """Group index (gene) controlling this GGUF tensor, or None if the tensor is not
        genome-controlled (embeddings, norms, output head, rope factors, ...)."""
        m = _BLK_RE.match(name)
        if m is None:
            return None
        layer, stem = int(m.group(1)), m.group(2)
        if stem not in self.linear_tensors or layer >= self.n_layers:
            return None
        return self.group_of_layer(layer)

    def reference_genome(self) -> list[Precision]:
        return [self.reference_precision] * N_SUPER_BLOCKS

    def baselines(self) -> dict[str, list[Precision]]:
        """Baseline genomes restricted to this model's alphabet: one uniform genome per
        precision, plus the published 'quantize middle layers harder' heuristic with its three
        grades (outer / next / inner) mapped onto the alphabet from highest to lowest precision."""
        levels = sorted(self.alphabet, key=lambda p: -p.bits)
        out = {f"uniform_{p.value}": [p] * N_SUPER_BLOCKS for p in levels}
        grade = [0, 1, 1, 2, 2, 1, 1, 0]
        out["heuristic"] = [levels[min(g, len(levels) - 1)] for g in grade]
        return out


_LLAMA_LINEARS = (
    "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "phi3-mini": ModelSpec(
        key="phi3-mini",
        hf_repo="microsoft/Phi-3-mini-4k-instruct",
        n_layers=32,
        # Fused projections: attn_qkv = qkv_proj, ffn_up = gate_up_proj.
        linear_tensors=("attn_qkv", "attn_output", "ffn_up", "ffn_down"),
        params_b=3.8,
    ),
    "llama3.2-3b": ModelSpec(
        key="llama3.2-3b",
        hf_repo="meta-llama/Llama-3.2-3B-Instruct",
        n_layers=28,
        linear_tensors=_LLAMA_LINEARS,
        params_b=3.2,
        gated=True,
    ),
    "llama3.1-8b": ModelSpec(
        key="llama3.1-8b",
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        n_layers=32,
        linear_tensors=_LLAMA_LINEARS,
        # F16 is ~16 GB: tight on the M4's 24 GiB, impossible on the 10 GB 3080.
        reference_precision=Precision.INT8,
        fixed_precision=Precision.INT8,
        alphabet=(Precision.INT8, Precision.INT4),
        params_b=8.0,
        gated=True,
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(MODEL_SPECS)}") from None
