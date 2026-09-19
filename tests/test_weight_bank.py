"""Unit tests for the Phase 2 weight bank (evol_inference/weight_bank.py). Shared
`tokenizer`/`model`/`bank` fixtures live in conftest.py."""

import torch

from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision


@torch.no_grad()
def _forward_logits(model, tokenizer):
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
    return model(**inputs).logits


def test_initial_active_state_is_all_fp16(bank):
    assert bank.active == [Precision.FP16] * N_SUPER_BLOCKS


def test_all_fp16_genome_exactly_reproduces_baseline(bank, model, tokenizer):
    baseline_logits = _forward_logits(model, tokenizer)
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    logits = _forward_logits(model, tokenizer)
    assert torch.equal(logits, baseline_logits)


def test_mixed_genome_updates_active_state(bank):
    genome = [Precision.INT8, Precision.INT4] * (N_SUPER_BLOCKS // 2)
    bank.assemble(genome)
    assert bank.active == genome


def test_reassembling_same_genome_is_a_noop(bank):
    genome = [Precision.INT8] * N_SUPER_BLOCKS
    bank.assemble(genome)
    first_modules = [bank._bank[i][(0, "mlp.down_proj")].int8 for i in range(N_SUPER_BLOCKS)]
    bank.assemble(genome)  # same genome again
    second_modules = [bank._bank[i][(0, "mlp.down_proj")].int8 for i in range(N_SUPER_BLOCKS)]
    assert first_modules == second_modules


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
