"""ModelSpec grouping / tensor classification -- pure logic, runs anywhere (tier T0)."""

import pytest

from evol_inference.genome import N_SUPER_BLOCKS
from evol_inference.model_spec import MODEL_SPECS, ModelSpec, get_spec, group_sizes


@pytest.mark.parametrize("n_layers", [8, 24, 26, 28, 32, 36, 40])
def test_group_sizes_partition_all_layers_near_equally(n_layers):
    sizes = group_sizes(n_layers)
    assert len(sizes) == N_SUPER_BLOCKS
    assert sum(sizes) == n_layers
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


def test_group_sizes_examples():
    assert group_sizes(32) == [4] * 8
    assert group_sizes(28) == [4, 4, 4, 4, 3, 3, 3, 3]


def test_group_sizes_rejects_too_few_layers():
    with pytest.raises(ValueError):
        group_sizes(7)


@pytest.mark.parametrize("key", sorted(MODEL_SPECS))
def test_every_layer_maps_to_exactly_one_group_in_order(key):
    spec = get_spec(key)
    groups = [spec.group_of_layer(i) for i in range(spec.n_layers)]
    assert groups == sorted(groups)
    assert set(groups) == set(range(N_SUPER_BLOCKS))
    for g in range(N_SUPER_BLOCKS):
        assert [spec.group_of_layer(i) for i in spec.layers_of_group(g)] == [g] * spec.sizes[g]


def test_group_of_layer_out_of_range():
    with pytest.raises(IndexError):
        get_spec("phi3-mini").group_of_layer(32)


def test_classify_tensor_phi3():
    spec = get_spec("phi3-mini")
    assert spec.classify_tensor("blk.0.attn_qkv.weight") == 0
    assert spec.classify_tensor("blk.3.ffn_down.weight") == 0
    assert spec.classify_tensor("blk.4.ffn_up.weight") == 1
    assert spec.classify_tensor("blk.31.attn_output.weight") == 7


@pytest.mark.parametrize(
    "name",
    ["token_embd.weight", "output.weight", "output_norm.weight", "blk.0.attn_norm.weight",
     "blk.0.attn_qkv.bias", "blk.99.attn_qkv.weight", "rope_factors_long.weight"],
)
def test_classify_tensor_ignores_non_genome_tensors(name):
    assert get_spec("phi3-mini").classify_tensor(name) is None


def test_llama_specs_use_separate_projections():
    for key in ("llama3.2-3b", "llama3.1-8b"):
        spec = get_spec(key)
        assert {"attn_q", "attn_k", "attn_v", "ffn_gate"} <= set(spec.linear_tensors)
    assert get_spec("llama3.2-3b").sizes == [4, 4, 4, 4, 3, 3, 3, 3]


def test_reference_precision_8b_is_int8_because_fp16_does_not_fit():
    from evol_inference.genome import Precision

    assert get_spec("llama3.1-8b").reference_precision is Precision.INT8
    assert get_spec("phi3-mini").reference_precision is Precision.FP16


def test_unknown_model_lists_known():
    with pytest.raises(KeyError, match="phi3-mini"):
        get_spec("nope")


def test_llama_8b_alphabet_excludes_fp16_and_baselines_follow_it():
    from evol_inference.genome import Precision

    spec = get_spec("llama3.1-8b")
    assert Precision.FP16 not in spec.alphabet
    b = spec.baselines()
    assert set(b) == {"uniform_int8", "uniform_int4", "heuristic"}
    assert all(set(g) <= set(spec.alphabet) for g in b.values())
    assert b["heuristic"] == [Precision.INT8, Precision.INT4, Precision.INT4, Precision.INT4,
                              Precision.INT4, Precision.INT4, Precision.INT4, Precision.INT8]


def test_phi3_baselines_match_the_original_four():
    from evol_inference.baselines import BASELINES

    b = get_spec("phi3-mini").baselines()
    assert b["heuristic"] == BASELINES["heuristic"]
    assert b["uniform_int8"] == BASELINES["uniform_int8"] and b["uniform_fp16"] == BASELINES["fp16"]
