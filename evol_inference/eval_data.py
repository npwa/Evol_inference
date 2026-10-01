"""Evaluation text for the llama.cpp accuracy probe.

The base project evaluates on the first 2048 tokens of WikiText-2's concatenated test split
(`fitness.load_fixed_eval_tokens`). llama.cpp's `llama-perplexity` reads a text file, and
this writes exactly the same concatenation. Verified (T1 step 4): Phi-3's HF tokenizer and
llama.cpp's tokenizer give identical token IDs over the whole file (341,468 tokens).

Scored window, stated plainly: `llama-perplexity -c 2048 --chunks 1` scores only the *second
half* of the chunk (1023 tokens, each with >=1024 tokens of context) -- not all 2047 as the
HF path does. The two paths therefore report different absolute perplexities; on the
same window they agree (HF 4.5229 vs llama.cpp 4.5221 on FP16 Phi-3, 0.02%). Never
compare absolute numbers across the two windows; fitness uses penalties relative to the
same backend's own baseline, so it is unaffected."""

from __future__ import annotations

from pathlib import Path

N_EVAL_TOKENS = 2048
# tokens scored by llama-perplexity for one chunk of n_ctx: positions n_ctx//2 .. n_ctx-2
def scored_tokens(n_ctx: int = N_EVAL_TOKENS, chunks: int = 1) -> int:
    return chunks * (n_ctx - 1 - n_ctx // 2)


def write_wikitext2_test(path: str | Path) -> Path:
    """Write WikiText-2 (raw) test split, joined exactly as the HF path joins it. Idempotent."""
    path = Path(path)
    if not path.exists():
        from datasets import load_dataset

        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return path
