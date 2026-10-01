"""Tier T1: real llama.cpp binaries + real Phi-3-mini GGUFs (Doc/implementation_plan_mac-m4.md
§3.3, §15). Skipped unless a built llama.cpp and the converted F16 GGUF are present:

    LLAMA_CPP_DIR   llama.cpp checkout with build-cpu/ (default ~/work/llama.cpp)
    PHI3_F16_GGUF   output of convert_hf_to_gguf.py --outtype f16 (default models/gguf/phi3-mini-f16.gguf)

Uniform source GGUFs are produced with `llama-quantize --pure` (no k-quant mixtures, so
every tensor of a file has exactly the requested type) and cached next to the F16 file.
"""

import os
import subprocess
from pathlib import Path

import gguf
import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.gguf_assembler import GgufAssembler, tensor_hashes
from evol_inference.model_spec import GGML_TYPE_NAME, get_spec

pytestmark = pytest.mark.llamacpp

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp")).expanduser()
QUANTIZE = LLAMA / "build-cpu/bin/llama-quantize"
F16 = Path(os.environ.get("PHI3_F16_GGUF", "models/gguf/phi3-mini-f16.gguf"))
SPEC = get_spec("phi3-mini")

if not (QUANTIZE.exists() and F16.exists()):
    pytest.skip("needs a built llama.cpp (build-cpu) and the converted Phi-3 F16 GGUF", allow_module_level=True)

GENOMES = {
    "edges_fp16": [Precision.FP16] + [Precision.INT8] * 6 + [Precision.FP16],
    "mixed": [Precision.FP16, Precision.INT8, Precision.INT4, Precision.INT8,
              Precision.INT4, Precision.INT4, Precision.INT8, Precision.FP16],
}


def quantize(src: Path, dst: Path, ftype: str, *extra: str) -> Path:
    if not dst.exists():
        subprocess.run([str(QUANTIZE), *extra, str(src), str(dst), ftype, "8"], check=True,
                       capture_output=True)
    return dst


@pytest.fixture(scope="module")
def sources():
    d = F16.parent
    return {
        Precision.FP16: F16,
        Precision.INT8: quantize(F16, d / "phi3-mini-q8_0-pure.gguf", "Q8_0", "--pure"),
        Precision.INT4: quantize(F16, d / "phi3-mini-q4_0-pure.gguf", "Q4_0", "--pure"),
    }


def test_real_tensor_names_match_model_spec():
    r = gguf.GGUFReader(F16)
    names = {t.name for t in r.tensors}
    assert r.get_field("general.architecture").contents() == "phi3"
    assert r.get_field("phi3.block_count").contents() == SPEC.n_layers
    for layer in range(SPEC.n_layers):
        for n in SPEC.genome_tensor_names(layer):
            assert n in names, n
    # every blk.* tensor not in the genome table is a norm; nothing genome-sized is missed
    other = {n for n in names if n.startswith("blk.") and SPEC.classify_tensor(n) is None}
    assert all(n.endswith("_norm.weight") for n in other), sorted(other)[:5]


def test_pure_sources_have_the_requested_type_on_every_genome_tensor(sources):
    for p, path in sources.items():
        want = getattr(gguf.GGMLQuantizationType, GGML_TYPE_NAME[p])
        for t in gguf.GGUFReader(path).tensors:
            if SPEC.classify_tensor(t.name) is not None:
                assert t.tensor_type == want, (p, t.name, t.tensor_type)


@pytest.mark.parametrize("name", sorted(GENOMES))
def test_assembled_genome_is_bit_identical_to_llama_quantize_tensor_type(sources, tmp_path, name):
    """The central claim of the assembler: copying pre-quantized tensors gives exactly what
    llama.cpp's own mixed quantization (per-tensor type overrides) produces."""
    genome = GENOMES[name]
    assembled = GgufAssembler(SPEC, sources).assemble(genome, tmp_path / "assembled.gguf")

    overrides = []
    for g, p in enumerate(genome):
        layers = "|".join(str(i) for i in SPEC.layers_of_group(g))
        stems = "|".join(SPEC.linear_tensors)
        overrides += ["--tensor-type", rf"blk\.({layers})\.({stems})\.weight={GGML_TYPE_NAME[p].lower()}"]
    reference = quantize(F16, tmp_path / "reference.gguf", "Q8_0", "--pure", *overrides)

    got, ref = tensor_hashes(assembled), tensor_hashes(reference)
    assert set(got) == set(ref)
    bad = sorted(n for n in got if got[n] != ref[n])
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:5]}"
    types = {t.name: t.tensor_type for t in gguf.GGUFReader(reference).tensors}
    for t in gguf.GGUFReader(assembled).tensors:
        assert t.tensor_type == types[t.name], t.name
