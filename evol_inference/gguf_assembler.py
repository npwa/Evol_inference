"""GGUF genome assembler: the llama.cpp-side replacement for `WeightBank`'s module
substitution (Doc/implementation_plan_mac-m4.md §3.3).

One uniform GGUF per precision is produced once with `llama-quantize` (same imatrix, if
any, for all). A genome is then *assembled* by copying each genome-controlled tensor's
already-quantized bytes from the source file its super-block's gene selects. Quantization
is per tensor, so a tensor's bytes are identical whether it came from a uniform or a
mixed file -- assembly is exact, and takes seconds instead of a full re-quantization.
Tensors the genome does not control (embeddings, output head, norms) come from the
`fixed` source file.

Needs only `gguf` (pip) -- no llama.cpp binary -- so this runs and is tested on any host.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import gguf
import numpy as np

from evol_inference.genome import Genome, N_SUPER_BLOCKS, Precision
from evol_inference.model_spec import ModelSpec


class AssemblyError(ValueError):
    pass


class GgufAssembler:
    def __init__(self, spec: ModelSpec, sources: dict[Precision, str | Path]):
        """`sources`: precision -> uniform GGUF path. Must contain every precision the
        genomes will use plus `spec.fixed_precision`."""
        self.spec = spec
        self.paths = {p: Path(path) for p, path in sources.items()}
        if spec.fixed_precision not in self.paths:
            raise AssemblyError(f"missing source for fixed precision {spec.fixed_precision}")
        self._readers = {p: gguf.GGUFReader(path) for p, path in self.paths.items()}
        self._tensors = {
            p: {t.name: t for t in r.tensors} for p, r in self._readers.items()
        }
        self._validate()

    def _validate(self) -> None:
        fixed = self._tensors[self.spec.fixed_precision]
        for p, tensors in self._tensors.items():
            if set(tensors) != set(fixed):
                diff = sorted(set(tensors) ^ set(fixed))[:5]
                raise AssemblyError(f"{p.value} source has a different tensor set, e.g. {diff}")
            for name, t in tensors.items():
                if list(t.shape) != list(fixed[name].shape):
                    raise AssemblyError(f"shape mismatch for {name} in {p.value} source")
        # Every genome-controlled tensor the spec predicts must exist.
        for layer in range(self.spec.n_layers):
            for name in self.spec.genome_tensor_names(layer):
                if name not in fixed:
                    raise AssemblyError(
                        f"{name} not in GGUF: wrong ModelSpec for this file? "
                        f"(spec {self.spec.key}, {self.spec.n_layers} layers)"
                    )

    def source_for(self, tensor_name: str, genome: Genome) -> Precision:
        gene = self.spec.classify_tensor(tensor_name)
        return self.spec.fixed_precision if gene is None else genome[gene]

    def assemble(self, genome: Genome, out_path: str | Path) -> Path:
        if len(genome) != N_SUPER_BLOCKS:
            raise AssemblyError(f"genome must have {N_SUPER_BLOCKS} genes, got {len(genome)}")
        missing = {p for p in genome if p not in self.paths}
        if missing:
            raise AssemblyError(f"no source GGUF for {sorted(p.value for p in missing)}")

        out_path = Path(out_path)
        fixed_reader = self._readers[self.spec.fixed_precision]
        arch = fixed_reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
        writer = gguf.GGUFWriter(out_path, arch)
        for field in fixed_reader.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue  # written by the writer itself
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

        plan = [
            (t.name, self._tensors[self.source_for(t.name, genome)][t.name])
            for t in fixed_reader.tensors
        ]
        for name, t in plan:
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for _, t in plan:
            writer.write_tensor_data(t.data)
        writer.close()
        return out_path

    def super_block_param_counts(self) -> list[int]:
        """Parameters in the genome-controlled tensors of each super-block (the `s_i` of
        the README's separability analysis; unequal for uneven groupings)."""
        fixed = self._tensors[self.spec.fixed_precision]
        counts = [0] * N_SUPER_BLOCKS
        for name, t in fixed.items():
            gene = self.spec.classify_tensor(name)
            if gene is not None:
                counts[gene] += int(t.n_elements)
        return counts

    def bytes_for(self, genome: Genome) -> int:
        """Exact on-disk bytes of the genome-controlled tensors under `genome` (not the
        analytical bits/8 proxy: includes block-scale overhead)."""
        total = 0
        for name, t in self._tensors[self.spec.fixed_precision].items():
            gene = self.spec.classify_tensor(name)
            if gene is not None:
                total += int(self._tensors[genome[gene]][name].n_bytes)
        return total


def tensor_hashes(path: str | Path) -> dict[str, str]:
    """sha256 of each tensor's raw data, keyed by tensor name -- used to prove an
    assembled file is bit-identical, tensor for tensor, to a reference (e.g. llama.cpp's
    own mixed quantization via `llama-quantize --tensor-type`)."""
    r = gguf.GGUFReader(path)
    return {t.name: hashlib.sha256(np.ascontiguousarray(t.data).tobytes()).hexdigest() for t in r.tensors}


class AssembledGgufProvider:
    """`genome -> GGUF path` callable for the llama.cpp probes: assembles into one reused work
    file (so a search never accumulates 2-7 GB files) and skips the rewrite when the genome
    is the same as the last one requested. Put `work_path` on a fast disk (tmpfs/NVMe); the
    OS page cache then makes the subsequent `llama-*` load cheap."""

    def __init__(self, assembler: GgufAssembler, work_path: str | Path):
        self.assembler = assembler
        self.work_path = Path(work_path)
        self._last: tuple[str, ...] | None = None
        self.n_assemblies = 0

    def __call__(self, genome: Genome) -> Path:
        key = tuple(p.value for p in genome)
        if key != self._last:
            self._last = None  # the file is invalid until assembly finishes
            self.assembler.assemble(genome, self.work_path)
            self._last = key
            self.n_assemblies += 1
        return self.work_path
