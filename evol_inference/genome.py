"""Genome definition shared by every backend: 8 genes, one per super-block, each a
`Precision`. Lives apart from `weight_bank.py` so that importing a genome (GA, baselines,
fitness, assembler) never pulls in bitsandbytes, which has no Arm/macOS build.

`Precision` keeps its original member names/values (`fp16`/`int8`/`int4`) because they
are the genome-cache and snapshot keys; non-bitsandbytes backends map them to their own
formats (GGUF: F16 / Q8_0 / Q4_0, see `evol_inference/model_spec.py`)."""

from __future__ import annotations

from enum import Enum

N_TRANSFORMER_BLOCKS = 32  # Phi-3-mini; other models: see ModelSpec.n_layers
N_SUPER_BLOCKS = 8  # the genome is always 8 genes, for every model
BLOCKS_PER_SUPER = N_TRANSFORMER_BLOCKS // N_SUPER_BLOCKS


class Precision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

    @property
    def bits(self) -> int:
        return {"fp16": 16, "int8": 8, "int4": 4}[self.value]


Genome = list[Precision]
