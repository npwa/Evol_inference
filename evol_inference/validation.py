"""Phase 7: broader multi-dataset perplexity validation (Doc/requirements.md §5,
Doc/implementation_plan.md Phase 7).

Runs once, after the search, on the baselines and the GA's final surviving population
only -- cheap, since it's a handful of configs, not every individual the search ever
touched. Confirms the fitness-loop's single-dataset (WikiText-2) accuracy signal isn't
an artifact of overfitting to that one dataset's quirks, by checking the same genomes
against two more, deliberately different-domain datasets: C4 (generic web text) and
Lambada (narrative-text word-prediction) -- see requirements.md §5 for why Lambada
replaced the originally-planned PTB (broken dataset loader).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from datasets import load_dataset

from evol_inference.fitness import compute_perplexity, genome_key
from evol_inference.weight_bank import Genome, WeightBank

# Same fixed-subset-size principle as the fitness loop (§6 determinism note).
N_VALIDATION_TOKENS = 2048
N_C4_EXAMPLES = 50  # C4 is enormous and streamed; this many examples comfortably covers N_VALIDATION_TOKENS


def load_wikitext2_tokens(tokenizer, n_tokens: int = N_VALIDATION_TOKENS, device: str = "cuda") -> torch.Tensor:
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)


def load_c4_tokens(tokenizer, n_tokens: int = N_VALIDATION_TOKENS, device: str = "cuda") -> torch.Tensor:
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    text = "\n\n".join(ex["text"] for ex in itertools.islice(ds, N_C4_EXAMPLES))
    return tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)


def load_lambada_tokens(tokenizer, n_tokens: int = N_VALIDATION_TOKENS, device: str = "cuda") -> torch.Tensor:
    ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)


@dataclass(frozen=True)
class ValidationResult:
    wikitext2_perplexity: float
    c4_perplexity: float
    lambada_perplexity: float


class ValidationSuite:
    """Evaluates genomes' perplexity across all three datasets, cached by genome (same
    rationale as FitnessEvaluator, §6) -- cheap insurance in case a genome shows up in
    both the baselines and the GA's final population."""

    def __init__(self, bank: WeightBank, tokenizer, n_tokens: int = N_VALIDATION_TOKENS):
        self.bank = bank
        self.wikitext2_tokens = load_wikitext2_tokens(tokenizer, n_tokens, bank.device)
        self.c4_tokens = load_c4_tokens(tokenizer, n_tokens, bank.device)
        self.lambada_tokens = load_lambada_tokens(tokenizer, n_tokens, bank.device)
        self._cache: dict[tuple[str, ...], ValidationResult] = {}

    def evaluate(self, genome: Genome) -> ValidationResult:
        key = genome_key(genome)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self.bank.assemble(genome)
        result = ValidationResult(
            wikitext2_perplexity=compute_perplexity(self.bank.model, self.wikitext2_tokens),
            c4_perplexity=compute_perplexity(self.bank.model, self.c4_tokens),
            lambada_perplexity=compute_perplexity(self.bank.model, self.lambada_tokens),
        )
        self._cache[key] = result
        return result
