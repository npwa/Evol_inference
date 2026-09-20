"""Unit tests for the Phase 7 validation harness (evol_inference/validation.py)."""

import pytest

from evol_inference.validation import (
    ValidationSuite,
    load_c4_tokens,
    load_lambada_tokens,
    load_wikitext2_tokens,
)
from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision


def test_load_wikitext2_tokens_returns_requested_length(tokenizer):
    assert load_wikitext2_tokens(tokenizer, n_tokens=64, device="cuda").shape == (1, 64)


def test_load_c4_tokens_returns_requested_length(tokenizer):
    assert load_c4_tokens(tokenizer, n_tokens=64, device="cuda").shape == (1, 64)


def test_load_lambada_tokens_returns_requested_length(tokenizer):
    assert load_lambada_tokens(tokenizer, n_tokens=64, device="cuda").shape == (1, 64)


@pytest.fixture(scope="session")
def validation_suite(bank, tokenizer):
    return ValidationSuite(bank, tokenizer, n_tokens=64)  # small for fast tests


def test_evaluate_returns_three_positive_perplexities(validation_suite):
    result = validation_suite.evaluate([Precision.FP16] * N_SUPER_BLOCKS)
    assert result.wikitext2_perplexity > 0
    assert result.c4_perplexity > 0
    assert result.lambada_perplexity > 0


def test_cache_hit_avoids_recomputation(validation_suite, monkeypatch):
    genome = [Precision.INT8] * N_SUPER_BLOCKS
    first = validation_suite.evaluate(genome)

    import evol_inference.validation as validation_mod

    calls = []
    original = validation_mod.compute_perplexity

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(validation_mod, "compute_perplexity", spy)

    second = validation_suite.evaluate(genome)  # same genome again -- must be a cache hit
    assert second is first
    assert len(calls) == 0
