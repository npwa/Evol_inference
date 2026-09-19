"""Phase 2: per-layer quantized weight bank + genome assembly (Doc/requirements.md §6,
Doc/implementation_plan.md Phase 2).

Precomputes FP16/INT8/INT4 variants of every transformer block's quantizable linear
layers once, keeps the currently-unused variants on CPU, and assembles a specific genome
into the live GPU model by **module substitution** (swapping which `bitsandbytes.nn`
module a block's attribute points at) rather than tensor mutation. This is what
guarantees real low-bit bitsandbytes GEMM kernels run the forward pass, instead of
silently falling back to a dequantize-then-FP16-matmul path (see the Phase 2 note in
implementation_plan.md for why that distinction matters).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from enum import Enum

import bitsandbytes as bnb
import torch
from torch import nn

N_TRANSFORMER_BLOCKS = 32
N_SUPER_BLOCKS = 8
BLOCKS_PER_SUPER = N_TRANSFORMER_BLOCKS // N_SUPER_BLOCKS
LINEAR_NAMES = (
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


class Precision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bits(self) -> int:
        return {"fp16": 16, "int8": 8, "int4": 4}[self.value]


Genome = list[Precision]


def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _move(module: nn.Module, device: str) -> nn.Module:
    """Move a layer variant between devices.

    `bnb.nn.Linear8bitLt` needs special handling: bitsandbytes attaches `CB`/`SCB`
    directly to the `Int8Params` weight object as plain attributes, outside the `.data`
    mechanism that `nn.Module`'s generic `_apply()`/`.to()` machinery manages. Once a
    Linear8bitLt layer has already been quantized once, a plain `module.to(device)`
    reassigns `.weight.data` correctly but silently leaves `.weight.CB`/`.weight.SCB`
    stuck on their original device forever (confirmed empirically — repeated device
    round-trips this way monotonically grow VRAM usage without ever releasing it).
    Reassigning `.weight` itself (not just its `.data`) routes through `Int8Params.to()`
    properly, which does relocate `CB`/`SCB`; `.state.CB`/`.state.SCB` (a separate
    forward-pass cache on the module) also need clearing, since they're not touched by
    a parameter-level move.

    `Params4bit` (the INT4 path) doesn't have this problem — its `.to()` mutates
    `quant_state` in place, and a plain module-level `.to(device)` relocates it
    correctly (confirmed empirically), so it goes through the standard path below."""
    if isinstance(module, bnb.nn.Linear8bitLt):
        module.weight = module.weight.to(device)
        if module.bias is not None:
            module.bias = module.bias.to(device)
        module.state.CB = None
        module.state.SCB = None
        return module
    return module.to(device)


def _quantize_int8(fp16_linear: nn.Linear, device: str) -> bnb.nn.Linear8bitLt:
    layer = bnb.nn.Linear8bitLt(
        fp16_linear.in_features,
        fp16_linear.out_features,
        bias=fp16_linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    layer.weight = bnb.nn.Int8Params(
        fp16_linear.weight.data.clone(), requires_grad=False, has_fp16_weights=False
    )
    if fp16_linear.bias is not None:
        layer.bias = nn.Parameter(fp16_linear.bias.data.clone(), requires_grad=False)
    return _move(layer, device)  # quantization is triggered by this device move


def _quantize_int4(fp16_linear: nn.Linear, device: str) -> bnb.nn.Linear4bit:
    layer = bnb.nn.Linear4bit(
        fp16_linear.in_features,
        fp16_linear.out_features,
        bias=fp16_linear.bias is not None,
        compute_dtype=torch.float16,
        quant_type="nf4",
    )
    layer.weight = bnb.nn.Params4bit(
        fp16_linear.weight.data.clone(), requires_grad=False, quant_type="nf4"
    )
    if fp16_linear.bias is not None:
        layer.bias = nn.Parameter(fp16_linear.bias.data.clone(), requires_grad=False)
    return _move(layer, device)  # quantization is triggered by this device move


@dataclass
class _LayerVariants:
    fp16: nn.Linear
    int8: bnb.nn.Linear8bitLt
    int4: bnb.nn.Linear4bit

    def get(self, precision: Precision) -> nn.Module:
        return getattr(self, precision.value)

    def set(self, precision: Precision, module: nn.Module) -> None:
        setattr(self, precision.value, module)


class WeightBank:
    """Precomputed FP16/INT8/INT4 variants of every quantizable linear layer, grouped
    into the 8 super-blocks from requirements.md §4, with genome assembly via module
    substitution against a live model."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        # {super_block_idx: {(block_offset, linear_name): _LayerVariants}}
        self._bank: dict[int, dict[tuple[int, str], _LayerVariants]] = {}
        self._active: Genome = [Precision.FP16] * N_SUPER_BLOCKS
        self._build()

    def _iter_layers(self):
        for super_idx in range(N_SUPER_BLOCKS):
            for block_offset in range(BLOCKS_PER_SUPER):
                block_idx = super_idx * BLOCKS_PER_SUPER + block_offset
                block = self.model.model.layers[block_idx]
                for linear_name in LINEAR_NAMES:
                    yield super_idx, block_offset, block, linear_name

    def _build(self) -> None:
        """Quantize every layer once. Each layer is quantized on GPU (bitsandbytes
        requires a CUDA device move to run quantization) then immediately relocated to
        CPU before the next layer, so peak VRAM during the build never exceeds the
        already-loaded FP16 baseline plus one layer's transient quantization buffer."""
        for super_idx, block_offset, block, linear_name in self._iter_layers():
            parent, attr = _get_parent_and_attr(block, linear_name)
            fp16_linear: nn.Linear = getattr(parent, attr)

            int8_linear = _move(_quantize_int8(fp16_linear, self.device), "cpu")
            int4_linear = _move(_quantize_int4(fp16_linear, self.device), "cpu")

            self._bank.setdefault(super_idx, {})[(block_offset, linear_name)] = (
                _LayerVariants(fp16=fp16_linear, int8=int8_linear, int4=int4_linear)
            )
            # bitsandbytes' custom Parameter subclasses (Int8Params/Params4bit) can
            # leave reference cycles behind after a device move that plain refcounting
            # won't collect; empty_cache() alone won't free memory still referenced by
            # an uncollected cycle, so gc.collect() has to run first.
            gc.collect()
            torch.cuda.empty_cache()

    def assemble(self, genome: Genome) -> None:
        """Reassign each super-block's linear layers to the genome's chosen precision,
        moving the newly-active variant to GPU and the previously-active one back to
        CPU. A super-block whose gene matches its current state is left untouched.

        Within a super-block, every old-precision layer is detached from the live model
        and freed *before* the new-precision layers are brought onto GPU — allocating
        the new layers first (freeing the old ones only afterward) can transiently need
        both the old and new super-block resident on GPU at once, which is enough to
        exceed a 10GB budget on a genome-wide jump toward FP16."""
        assert len(genome) == N_SUPER_BLOCKS
        for super_idx, target in enumerate(genome):
            current = self._active[super_idx]
            if target == current:
                continue

            layers = list(self._bank[super_idx].items())

            # Detach this super-block's active layers from the live model and move
            # them back to CPU, freeing their GPU memory.
            detached = []
            for (block_offset, linear_name), variants in layers:
                block_idx = super_idx * BLOCKS_PER_SUPER + block_offset
                block = self.model.model.layers[block_idx]
                parent, attr = _get_parent_and_attr(block, linear_name)
                old_module = getattr(parent, attr)
                setattr(parent, attr, None)  # drop the model's reference before moving
                detached.append((variants, old_module))
            for variants, old_module in detached:
                variants.set(current, _move(old_module, "cpu"))
            del detached
            gc.collect()
            torch.cuda.empty_cache()

            # Now bring the target precision's layers onto GPU and attach them.
            for (block_offset, linear_name), variants in layers:
                block_idx = super_idx * BLOCKS_PER_SUPER + block_offset
                block = self.model.model.layers[block_idx]
                parent, attr = _get_parent_and_attr(block, linear_name)
                new_module = _move(variants.get(target), self.device)
                setattr(parent, attr, new_module)
                variants.set(target, new_module)

            self._active[super_idx] = target
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def active(self) -> Genome:
        return list(self._active)
