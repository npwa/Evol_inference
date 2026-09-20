"""Phase 3: fitness harness (Doc/requirements.md §5, Doc/implementation_plan.md Phase 3).

`accuracy_penalty` and `efficiency_gain` are each a fraction of the **fixed** FP16
baseline (not a percentile rank within the live population — an earlier draft had that
bug, see requirements.md §5 for why it broke steady-state caching). That fixed-baseline
normalization is what makes a genome's fitness a pure, deterministic function of the
genome, safe to cache forever by genome hash — which matters because the steady-state GA
re-derives the same genome often (mutation reverting a gene, crossover reconstructing an
already-seen individual, near-duplicates late in the search).

Real wall-clock latency is measured too, but only as a report-only diagnostic (§5/§6) —
it never feeds into the fitness score, precisely so fitness stays comparable across
whichever host evaluated a given genome.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import torch
from datasets import load_dataset

from evol_inference.weight_bank import Genome, N_SUPER_BLOCKS, Precision, WeightBank

# Fixed token subset (§6 determinism note): every fitness evaluation during the search
# runs against exactly the same tokens, every time, on every host.
N_EVAL_TOKENS = 2048


def load_fixed_eval_tokens(tokenizer, n_tokens: int = N_EVAL_TOKENS, device: str = "cuda") -> torch.Tensor:
    """The first `n_tokens` tokens of WikiText-2's concatenated test split — always the
    same slice, so results are comparable across genomes, hosts, and reruns.

    Uses `Salesforce/wikitext` rather than the unnamespaced `wikitext` repo: the latter
    hit a real `huggingface_hub` URI-parsing bug on this installed version (`HfUriError`
    on a pinned-revision path for single-segment repo ids), confirmed reproducible and
    unrelated to network/auth. `Salesforce/wikitext` is the same dataset, canonical
    parquet-backed reupload, and loads cleanly."""
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    return input_ids.to(device)


def genome_key(genome: Genome) -> tuple[str, ...]:
    """A hashable, JSON-friendly cache key for a genome."""
    return tuple(p.value for p in genome)


@torch.no_grad()
def compute_perplexity(model, input_ids: torch.Tensor) -> float:
    loss = model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()


@torch.no_grad()
def measure_latency_ms(model, input_ids: torch.Tensor, n_runs: int = 3, n_warmup: int = 1) -> float:
    for _ in range(n_warmup):
        model(input_ids)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        model(input_ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000.0


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    accuracy_penalty: float
    efficiency_gain: float
    perplexity: float
    bytes_: float
    latency_ms: float  # report-only (§5/§6) -- never feeds into `fitness`


class FitnessEvaluator:
    """Evaluates genomes against a WeightBank's live model. `accuracy_penalty` and
    `efficiency_gain` are normalized against fixed FP16-baseline constants computed once
    at construction time (§5), and results are cached by genome (§6) — the cache stores
    only this dataclass's scalar fields, never model weights or tensors."""

    def __init__(
        self,
        bank: WeightBank,
        tokenizer,
        w1: float = 0.5,
        w2: float = 0.5,
        n_eval_tokens: int = N_EVAL_TOKENS,
        measure_latency: bool = True,
    ):
        self.bank = bank
        self.w1 = w1
        self.w2 = w2
        self.measure_latency = measure_latency
        self._cache: dict[tuple[str, ...], FitnessResult] = {}

        self.eval_tokens = load_fixed_eval_tokens(tokenizer, n_eval_tokens, bank.device)
        self.param_counts = bank.super_block_param_counts()
        self.bytes_fp16 = sum(self.param_counts) * Precision.FP16.bits / 8

        bank.assemble([Precision.FP16] * N_SUPER_BLOCKS)
        self.baseline_perplexity = compute_perplexity(bank.model, self.eval_tokens)

    def bytes_for(self, genome: Genome) -> float:
        return sum(
            count * precision.bits / 8 for count, precision in zip(self.param_counts, genome)
        )

    def evaluate(self, genome: Genome) -> FitnessResult:
        key = genome_key(genome)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        self.bank.assemble(genome)
        perplexity = compute_perplexity(self.bank.model, self.eval_tokens)
        bytes_ = self.bytes_for(genome)

        accuracy_penalty = (perplexity - self.baseline_perplexity) / self.baseline_perplexity
        efficiency_gain = 1.0 - bytes_ / self.bytes_fp16
        fitness = efficiency_gain * self.w1 - accuracy_penalty * self.w2

        latency_ms = (
            measure_latency_ms(self.bank.model, self.eval_tokens)
            if self.measure_latency
            else float("nan")
        )

        result = FitnessResult(
            fitness=fitness,
            accuracy_penalty=accuracy_penalty,
            efficiency_gain=efficiency_gain,
            perplexity=perplexity,
            bytes_=bytes_,
            latency_ms=latency_ms,
        )
        self._cache[key] = result
        return result

    def __len__(self) -> int:
        """Number of distinct genomes evaluated (cache size) — mostly for testing."""
        return len(self._cache)


def backfill_latency(evaluator: FitnessEvaluator, genomes: list[Genome]) -> None:
    """Measure real wall-clock latency for exactly these already-evaluated genomes and
    update their cached result in place (`FitnessResult` is frozen, so this replaces
    the cache entry with a copy).

    Latency is report-only and never feeds fitness (§5/§6), so for a long search run
    it's wasteful to pay its ~5x-forward-pass cost on every single evaluation --
    `FitnessEvaluator(measure_latency=False)` skips it during the search, and this
    backfills it once, afterward, only for the handful of genomes that end up in a
    report (the final population, the baselines)."""
    for genome in genomes:
        key = genome_key(genome)
        result = evaluator._cache[key]
        evaluator.bank.assemble(genome)
        latency_ms = measure_latency_ms(evaluator.bank.model, evaluator.eval_tokens)
        evaluator._cache[key] = replace(result, latency_ms=latency_ms)
