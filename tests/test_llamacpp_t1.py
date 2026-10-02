"""Tier T1: real llama.cpp binaries + real GGUFs (Doc/implementation_plan_mac-m4.md §3.3, §15,
§16). Skipped unless a built llama.cpp is present; each model's tests are skipped unless its
converted F16 GGUF exists:

    LLAMA_CPP_DIR   llama.cpp checkout with build-cpu/ (default ~/work/llama.cpp)
    GGUF_DIR        directory with <model-key>-f16.gguf from convert_hf_to_gguf.py --outtype f16
                    (default models/gguf); keys: phi3-mini, llama3.2-3b, llama3.1-8b

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
GGUF_DIR = Path(os.environ.get("GGUF_DIR", "models/gguf"))
# kept for the step-4 probe tests below (Phi-3)
F16 = GGUF_DIR / "phi3-mini-f16.gguf"
SPEC = get_spec("phi3-mini")

if not QUANTIZE.exists():
    pytest.skip("needs a built llama.cpp (build-cpu)", allow_module_level=True)

MODELS = ["phi3-mini", "llama3.1-8b", "llama3.2-3b"]


def f16_path(key: str) -> Path:
    p = GGUF_DIR / f"{key}-f16.gguf"
    if not p.exists():
        pytest.skip(f"no converted F16 GGUF for {key} at {p}")
    return p


def genomes_for(spec):
    """Mixed genomes drawn from the model's alphabet (edge-protected, and an alternating one)."""
    hi, lo = sorted(spec.alphabet, key=lambda p: -p.bits)[0], sorted(spec.alphabet, key=lambda p: -p.bits)[-1]
    mid = sorted(spec.alphabet, key=lambda p: -p.bits)[len(spec.alphabet) // 2]
    return {
        "edges_hi": [hi] + [lo] * 6 + [hi],
        "mixed": [hi, mid, lo, mid, lo, lo, mid, hi],
    }


def quantize(src: Path, dst: Path, ftype: str, *extra: str) -> Path:
    if not dst.exists():
        subprocess.run([str(QUANTIZE), *extra, str(src), str(dst), ftype, "8"], check=True,
                       capture_output=True)
    return dst


def sources_for(spec) -> dict:
    f16 = f16_path(spec.key)
    d = f16.parent
    src = {Precision.FP16: f16,
           Precision.INT8: quantize(f16, d / f"{spec.key}-q8_0-pure.gguf", "Q8_0", "--pure"),
           Precision.INT4: quantize(f16, d / f"{spec.key}-q4_0-pure.gguf", "Q4_0", "--pure")}
    return src


@pytest.fixture(scope="module")
def sources():
    return sources_for(SPEC)


MODEL_PARAMS = [pytest.param("phi3-mini"), pytest.param("llama3.2-3b"),
                pytest.param("llama3.1-8b", marks=pytest.mark.slow)]


@pytest.mark.parametrize("key", MODEL_PARAMS)
def test_real_tensor_names_match_model_spec(key):
    spec = get_spec(key)
    r = gguf.GGUFReader(f16_path(key))
    names = {t.name for t in r.tensors}
    arch = r.get_field("general.architecture").contents()
    assert r.get_field(f"{arch}.block_count").contents() == spec.n_layers
    for layer in range(spec.n_layers):
        for n in spec.genome_tensor_names(layer):
            assert n in names, n
    # every blk.* tensor outside the genome table is a norm; nothing genome-sized is missed
    other = {n for n in names if n.startswith("blk.") and spec.classify_tensor(n) is None}
    assert all(n.endswith("_norm.weight") for n in other), sorted(other)[:5]


@pytest.mark.parametrize("key", MODEL_PARAMS)
def test_pure_sources_have_the_requested_type_on_every_genome_tensor(key):
    spec = get_spec(key)
    for p, path in sources_for(spec).items():
        want = getattr(gguf.GGMLQuantizationType, GGML_TYPE_NAME[p])
        for t in gguf.GGUFReader(path).tensors:
            if spec.classify_tensor(t.name) is not None:
                assert t.tensor_type == want, (key, p, t.name, t.tensor_type)


@pytest.mark.parametrize("key", MODEL_PARAMS)
@pytest.mark.parametrize("name", ["edges_hi", "mixed"])
def test_assembled_genome_is_bit_identical_to_llama_quantize_tensor_type(key, name, tmp_path):
    """The central claim of the assembler: copying pre-quantized tensors gives exactly what
    llama.cpp's own mixed quantization (per-tensor type overrides) produces -- for every
    architecture (fused qkv/gate_up in Phi-3, separate q/k/v + GQA in Llama)."""
    spec = get_spec(key)
    srcs = sources_for(spec)
    genome = genomes_for(spec)[name]
    assembled = GgufAssembler(spec, srcs).assemble(genome, tmp_path / "assembled.gguf")

    overrides = []
    for g, p in enumerate(genome):
        layers = "|".join(str(i) for i in spec.layers_of_group(g))
        stems = "|".join(spec.linear_tensors)
        overrides += ["--tensor-type", rf"blk\.({layers})\.({stems})\.weight={GGML_TYPE_NAME[p].lower()}"]
    reference = quantize(f16_path(key), tmp_path / "reference.gguf", GGML_TYPE_NAME[spec.fixed_precision],
                         "--pure", *overrides)

    got, ref = tensor_hashes(assembled), tensor_hashes(reference)
    assert set(got) == set(ref)
    bad = sorted(n for n in got if got[n] != ref[n])
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:5]}"
    types = {t.name: t.tensor_type for t in gguf.GGUFReader(reference).tensors}
    for t in gguf.GGUFReader(assembled).tensors:
        assert t.tensor_type == types[t.name], t.name


# ---- step 4: the real accuracy probe -----------------------------------------------------------

from evol_inference.eval_data import write_wikitext2_test  # noqa: E402
from evol_inference.gguf_assembler import AssembledGgufProvider  # noqa: E402
from evol_inference.llama_probes import LlamaPerplexityProbe  # noqa: E402

_CUDA_PPL = LLAMA / "build-cuda/bin/llama-perplexity"
_PPL_BIN = _CUDA_PPL if _CUDA_PPL.exists() else LLAMA / "build-cpu/bin/llama-perplexity"
_NGL = 99 if _CUDA_PPL.exists() else 0


@pytest.fixture(scope="module")
def probe(sources, tmp_path_factory):
    work = tmp_path_factory.mktemp("work") / "genome.gguf"
    prov = AssembledGgufProvider(GgufAssembler(SPEC, sources), work)
    text = write_wikitext2_test(F16.parent / "wikitext2_test.txt")
    return LlamaPerplexityProbe(_PPL_BIN, text, prov, n_gpu_layers=_NGL)


def test_real_probe_is_deterministic_and_orders_precisions(probe):
    fp16 = probe.perplexity([Precision.FP16] * N_SUPER_BLOCKS)
    assert probe.perplexity([Precision.FP16] * N_SUPER_BLOCKS) == fp16
    int8 = probe.perplexity([Precision.INT8] * N_SUPER_BLOCKS)
    int4 = probe.perplexity([Precision.INT4] * N_SUPER_BLOCKS)
    assert fp16 < int8 < int4
    # HF fp16 on the same window (tokens 1025..2047 of the identical token stream): 4.5229
    assert fp16 == pytest.approx(4.5229, rel=0.005)


def test_real_probe_mixed_genome_lies_between_its_uniform_bounds(probe):
    lo = probe.perplexity([Precision.FP16] * N_SUPER_BLOCKS)
    hi = probe.perplexity([Precision.INT4] * N_SUPER_BLOCKS)
    mixed = probe.perplexity(genomes_for(SPEC)["mixed"])
    assert lo < mixed < hi
