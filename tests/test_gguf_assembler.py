"""GGUF assembler against synthetic GGUF files (tier T0): no llama.cpp, no model needed.
Tiny fake 12-layer model (uneven groups 2,2,2,2,1,1,1,1) with real Q8_0/Q4_0/F16 data."""

import gguf
import numpy as np
import pytest
from gguf.quants import quantize

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.gguf_assembler import AssemblyError, GgufAssembler, tensor_hashes
from evol_inference.model_spec import ModelSpec

SPEC = ModelSpec(
    key="tiny", hf_repo="none", n_layers=12, linear_tensors=("attn_q", "ffn_up"),
    fixed_precision=Precision.INT8,
)
ROWS, COLS = 4, 64  # COLS a multiple of the 32-wide quant block
QTYPE = {Precision.FP16: gguf.GGMLQuantizationType.F16,
         Precision.INT8: gguf.GGMLQuantizationType.Q8_0,
         Precision.INT4: gguf.GGMLQuantizationType.Q4_0}


def _weights(spec=SPEC):
    rng = np.random.default_rng(0)
    w = {"token_embd.weight": rng.standard_normal((ROWS, COLS), dtype=np.float32)}
    for layer in range(spec.n_layers):
        for stem in (*spec.linear_tensors, "attn_norm"):
            w[f"blk.{layer}.{stem}.weight"] = rng.standard_normal((ROWS, COLS), dtype=np.float32)
    return w


def _write_uniform(path, weights, precision, drop=None):
    w = gguf.GGUFWriter(path, "llama")
    w.add_string("general.name", "tiny")
    w.add_uint32("llama.block_count", SPEC.n_layers)
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    for name, data in weights.items():
        if name == drop:
            continue
        if name.endswith("norm.weight"):  # norms are always F32 in real GGUFs
            w.add_tensor(name, data)
        elif precision is Precision.FP16:
            w.add_tensor(name, data.astype(np.float16))
        else:
            q = quantize(data, QTYPE[precision])
            w.add_tensor(name, q, raw_dtype=QTYPE[precision])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    d = tmp_path_factory.mktemp("gguf")
    weights = _weights()
    paths = {}
    for p in Precision:
        paths[p] = d / f"{p.value}.gguf"
        _write_uniform(paths[p], weights, p)
    return paths


@pytest.fixture(scope="module")
def asm(sources):
    return GgufAssembler(SPEC, sources)


def test_uniform_genome_reproduces_uniform_source_exactly(asm, sources, tmp_path):
    for p in Precision:
        out = asm.assemble([p] * N_SUPER_BLOCKS, tmp_path / f"u_{p.value}.gguf")
        got, ref = tensor_hashes(out), tensor_hashes(sources[p])
        # genome-controlled tensors match the uniform source; others come from fixed (INT8)
        for name, h in got.items():
            expect = ref if SPEC.classify_tensor(name) is not None else tensor_hashes(sources[Precision.INT8])
            assert h == expect[name], name


def test_mixed_genome_takes_each_tensor_from_its_genes_source(asm, sources, tmp_path):
    genome = [Precision.FP16, Precision.INT8, Precision.INT4, Precision.INT8,
              Precision.INT4, Precision.FP16, Precision.INT8, Precision.INT4]
    out = asm.assemble(genome, tmp_path / "mixed.gguf")
    got = tensor_hashes(out)
    ref = {p: tensor_hashes(sources[p]) for p in Precision}
    for name, h in got.items():
        gene = SPEC.classify_tensor(name)
        src = SPEC.fixed_precision if gene is None else genome[gene]
        assert h == ref[src][name], (name, gene, src)
    # uneven groups: layer 11 belongs to gene 7 (size-1 group), layer 0-1 to gene 0
    assert SPEC.group_of_layer(11) == 7 and SPEC.group_of_layer(1) == 0


def test_assembled_file_keeps_metadata_and_tensor_types(asm, tmp_path):
    genome = [Precision.INT4] * N_SUPER_BLOCKS
    r = gguf.GGUFReader(asm.assemble(genome, tmp_path / "m.gguf"))
    assert r.get_field("general.name").contents() == "tiny"
    assert r.get_field("general.architecture").contents() == "llama"
    assert r.get_field("tokenizer.ggml.tokens").contents() == ["a", "b", "c"]
    types = {t.name: t.tensor_type for t in r.tensors}
    assert types["blk.3.attn_q.weight"] == gguf.GGMLQuantizationType.Q4_0
    assert types["blk.3.attn_norm.weight"] == gguf.GGMLQuantizationType.F32
    assert types["token_embd.weight"] == gguf.GGMLQuantizationType.Q8_0


def test_bytes_for_orders_precisions_and_is_additive(asm):
    b = {p: asm.bytes_for([p] * N_SUPER_BLOCKS) for p in Precision}
    assert b[Precision.INT4] < b[Precision.INT8] < b[Precision.FP16]
    mixed = [Precision.INT4] * 4 + [Precision.INT8] * 4
    # genes are additive per super-block
    expect = 0
    for g, p in enumerate(mixed):
        uniform = [p] * N_SUPER_BLOCKS
        only_g = asm.bytes_for([p if i == g else Precision.INT8 for i in range(N_SUPER_BLOCKS)])
        expect += only_g - asm.bytes_for([Precision.INT8] * N_SUPER_BLOCKS)
    assert asm.bytes_for(mixed) == asm.bytes_for([Precision.INT8] * N_SUPER_BLOCKS) + expect


def test_super_block_param_counts_follow_uneven_groups(asm):
    counts = asm.super_block_param_counts()
    per_layer = len(SPEC.linear_tensors) * ROWS * COLS
    assert counts == [per_layer * s for s in SPEC.sizes]


def test_rejects_wrong_spec_for_file(sources):
    bad = ModelSpec(key="bad", hf_repo="none", n_layers=12, linear_tensors=("attn_qkv",))
    with pytest.raises(AssemblyError, match="not in GGUF"):
        GgufAssembler(bad, sources)


def test_rejects_mismatched_tensor_sets(tmp_path):
    weights = _weights()
    paths = {}
    for p in (Precision.INT8, Precision.INT4):
        paths[p] = tmp_path / f"{p.value}.gguf"
    _write_uniform(paths[Precision.INT8], weights, Precision.INT8)
    _write_uniform(paths[Precision.INT4], weights, Precision.INT4, drop="blk.5.ffn_up.weight")
    with pytest.raises(AssemblyError, match="different tensor set"):
        GgufAssembler(SPEC, paths)


def test_rejects_bad_genome_length_and_missing_source(asm, sources, tmp_path):
    with pytest.raises(AssemblyError, match="8 genes"):
        asm.assemble([Precision.INT8] * 3, tmp_path / "x.gguf")
    partial = GgufAssembler(SPEC, {Precision.INT8: sources[Precision.INT8]})
    with pytest.raises(AssemblyError, match="no source"):
        partial.assemble([Precision.INT4] * N_SUPER_BLOCKS, tmp_path / "x.gguf")
