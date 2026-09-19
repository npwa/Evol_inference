"""Unit tests for the Phase 2 weight bank (evol_inference/weight_bank.py)."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision, WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="weight bank requires a CUDA device"
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="module")
def model():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16, device_map="cuda"
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def baseline_logits(model, tokenizer):
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        return model(**inputs).logits.clone()


@pytest.fixture(scope="module")
def bank(model):
    return WeightBank(model, device="cuda")


def _forward_logits(model, tokenizer):
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        return model(**inputs).logits


def test_initial_active_state_is_all_fp16(bank):
    assert bank.active == [Precision.FP16] * N_SUPER_BLOCKS


def test_all_fp16_genome_exactly_reproduces_baseline(bank, model, tokenizer, baseline_logits):
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    logits = _forward_logits(model, tokenizer)
    assert torch.equal(logits, baseline_logits)


def test_mixed_genome_updates_active_state(bank):
    genome = [Precision.INT8, Precision.INT4] * (N_SUPER_BLOCKS // 2)
    bank.assemble(genome)
    assert bank.active == genome
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)  # reset for other tests


def test_reassembling_same_genome_is_a_noop(bank):
    genome = [Precision.INT8] * N_SUPER_BLOCKS
    bank.assemble(genome)
    first_modules = [
        bank._bank[i][(0, "mlp.down_proj")].int8 for i in range(N_SUPER_BLOCKS)
    ]
    bank.assemble(genome)  # same genome again
    second_modules = [
        bank._bank[i][(0, "mlp.down_proj")].int8 for i in range(N_SUPER_BLOCKS)
    ]
    assert first_modules == second_modules
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)  # reset for other tests


def test_all_int4_genome_uses_less_vram_than_fp16(bank, model, tokenizer):
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    torch.cuda.reset_peak_memory_stats()
    _forward_logits(model, tokenizer)
    fp16_vram = torch.cuda.max_memory_allocated()

    bank.assemble([Precision.INT4] * N_SUPER_BLOCKS)
    torch.cuda.reset_peak_memory_stats()
    _forward_logits(model, tokenizer)
    int4_vram = torch.cuda.max_memory_allocated()

    assert int4_vram < fp16_vram
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)  # reset for other tests
