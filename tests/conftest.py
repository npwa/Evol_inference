"""Shared fixtures for the evol_inference test suite. `tokenizer`/`model`/`bank` are
session-scoped since building the weight bank (quantizing all 128 layers) takes about a
minute — every test file that needs a live model reuses the same instance rather than
paying that cost again."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.genome import N_SUPER_BLOCKS, Precision

MODEL_DIR = "./models/phi-3-mini-4k-instruct"

# Fixtures that need the live Phi-3 model on a CUDA device (T1). Tests using any of
# them are auto-marked `gpu` and skipped on hosts without CUDA, so the rest of the suite
# (T0: pure-Python logic, GGUF assembler, energy meter, ...) runs anywhere.
_GPU_FIXTURES = {"tokenizer", "model", "bank"}


def pytest_collection_modifyitems(config, items):
    has_cuda = torch.cuda.is_available()
    skip = pytest.mark.skip(reason="requires a CUDA device (tier T1)")
    for item in items:
        if _GPU_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.gpu)
            if not has_cuda:
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
    from evol_inference.weight_bank import WeightBank  # bitsandbytes: GPU tier only

    return WeightBank(model, device="cuda")


@pytest.fixture(autouse=True)
def _reset_bank_to_fp16(request):
    """Every GPU test starts from a known-clean all-FP16 state, and resets back to it
    afterward, so tests can run in any order without leaking state between them. A
    no-op for tests that don't use the `bank` fixture."""
    if "bank" not in request.fixturenames:
        yield
        return
    bank = request.getfixturevalue("bank")
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
    yield
    bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
