"""Shared fixtures for the evol_inference test suite. `tokenizer`/`model`/`bank` are
session-scoped since building the weight bank (quantizing all 128 layers) takes about a
minute — every test file that needs a live model reuses the same instance rather than
paying that cost again."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision, WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"

collect_ignore_glob = [] if torch.cuda.is_available() else ["*"]


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="test suite requires a CUDA device")
        for item in items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


@pytest.fixture(scope="session")
def model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    m.eval()
    return m


@pytest.fixture(scope="session")
def bank(model):
    return WeightBank(model, device="cuda")


@pytest.fixture(autouse=True)
def _reset_bank_to_fp16(bank):
    """Every test starts from a known-clean all-FP16 state, and resets back to it
    afterward, so tests can run in any order without leaking state between them."""
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    yield
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
