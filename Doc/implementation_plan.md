# Implementation Plan: Evolutionary Search for Mixed-Precision LLM Quantization

Planning only — no code yet. Steps are phased so each phase produces something runnable
and checkable before the next one depends on it. Section references (§N) point at
`Doc/requirements.md`.

Local dev host (verified): RTX 3080, 10GB VRAM, driver 580.126.09 (CUDA 12.x capable),
Python 3.12.3, no conda/uv currently installed — plan assumes a plain `venv` + `pip`.

---

## 1. Phased steps

### Phase 0 — Environment
1. Create a venv (`python3 -m venv .venv`), install the package list below.
2. Install a CUDA-enabled PyTorch build matching the driver (cu121 or cu124 wheel).
3. Sanity check: `torch.cuda.is_available()`, `torch.cuda.get_device_name(0)` →
   `NVIDIA GeForce RTX 3080`, and that `bitsandbytes` imports without a CUDA setup error
   (it self-checks CUDA compatibility on import).

### Phase 1 — Model + baseline load
1. Download Phi-3-mini-4k-instruct (command below).
2. Load the FP16 baseline with `transformers`, run one forward pass, confirm VRAM usage
   matches the ≈7.6GB estimate from §3.
3. Identify and enumerate the 32 transformer blocks' quantizable linear layers
   programmatically (needed before the weight bank can be built) — confirm the count and
   naming matches what §4's "8 super-blocks of 4 blocks each" grouping assumes.

### Phase 2 — Per-layer weight bank (§6)
1. Implement calibration-free block-wise/absmax INT8 and INT4 quantization for one
   super-block's linear layers, using `bitsandbytes.functional` primitives directly
   (not the `from_pretrained(load_in_4bit=...)` path — see §3/§6 for why).
2. Extend to all 8 super-blocks; store all three precision variants (FP16/INT8/INT4) per
   super-block in CPU RAM.
3. Implement "assemble a genome": given an 8-gene array, copy the right precision variant
   of each super-block's weights into one live GPU model instance.
4. Verify correctness: an all-INT8 genome's output should be close to bitsandbytes'
   standard `load_in_8bit` output on the same input (same quantization scheme, different
   code path); an all-FP16 genome should exactly reproduce the FP16 baseline.

### Phase 3 — Fitness harness (§5)
1. WikiText-2 perplexity evaluation over a **fixed token subset** (needed for the
   determinism note in §6) — implement once, reuse for every genome.
2. Analytical `bytes(genome)` calculation from the genome's per-layer bit-widths.
3. `accuracy_penalty` / `efficiency_gain` / scalarized `fitness` per the fixed-baseline
   formulas in §5.
4. Genome-hash → fitness cache (SQLite is enough for v1's single-host case; scalar
   metadata only, per §6).
5. Real wall-clock latency benchmark (report-only metric, §6) — measured on GPU, logged
   alongside but never fed into fitness.

### Phase 4 — GA engine (§6)
1. Genome representation: 8-gene array over `{FP16, INT8, INT4}`.
2. Linear rank-weighted parent selection (`s = 1.5` default).
3. Crossover (single/multi-point) + mutation (tunable per-gene reassignment rate).
4. Steady-state population as a fixed-capacity sorted list (worst evicted on overflow),
   backed by the same SQLite store as the fitness cache for v1.
5. Worker loop tying it together: select → crossover/mutate → cache check → evaluate →
   insert/evict. On a single GPU host, this loop is effectively sequential (the GPU forward
   pass is the bottleneck regardless of "async" framing) — the async/multi-worker payoff
   materializes once a second GPU host is added, per §6's "Heterogeneous hardware" section.
   No design change needed then, just pointing a second worker process at the same store.

### Phase 5 — Baselines (§7)
1. FP16 (all-FP16 genome — already producible by Phase 2's assembly logic).
2. Uniform INT8, uniform INT4 (all-INT8 / all-INT4 genomes).
3. Heuristic baseline: quantize middle super-blocks more aggressively than first/last.

### Phase 6 — Run the search
1. Run the steady-state loop for a fixed evaluation budget. *(Open question — see §
   below: no discrete "generations" means the stopping rule needs to be picked explicitly,
   e.g. total evaluation count or wall-clock budget, rather than "N generations.")*
2. Snapshot the final population + baselines.

### Phase 7 — Validation + deliverables (§5, §8)
1. Run the broader WikiText-2 + C4 + PTB perplexity validation pass on baselines + the
   GA's final population only (cheap — a handful of configs, §5).
2. Compute the Pareto front, generate the two plots (analytical bytes vs. measured
   latency, §8).
3. Write the README/blog-post framing (§8).

---

## 2. Packages to install

**Core ML / quantization**
- `torch` — CUDA build matching the driver (cu121 or cu124 wheel; RTX 3080 is Ampere/sm_86,
  well supported by any recent build)
- `transformers` (≥4.41, first version with Phi-3 support)
- `accelerate` (device placement / `from_pretrained` device_map helpers)
- `bitsandbytes` (≥0.43 — this is what supplies the low-level int8/int4 quantize/dequantize
  primitives the per-layer weight bank is built from, §6)
- `sentencepiece` (Phi-3's tokenizer depends on it)
- `safetensors` (weight file format `from_pretrained` will pull)

**Data**
- `datasets` (WikiText-2, C4, PTB — all pulled from the HF Hub)
- `huggingface_hub` (model download CLI/API)

**GA / coordination**
- nothing beyond the standard library for v1 (`sqlite3`, `hashlib`, `random`, `dataclasses`
  are all stdlib) — see the open question below on whether SQLite stays sufficient once a
  second host joins

**Analysis / deliverables**
- `matplotlib` (Pareto plots)
- `pandas` (results tables — optional but convenient)
- `numpy`

**Dev**
- `pytest` (unit tests for the weight-bank assembly and fitness math — both are easy to get
  subtly wrong and cheap to test in isolation)

**Not needed for v1** (flagged so they're not installed speculatively): `auto-gptq`,
`auto-awq` (only relevant if the weight-bank approach in Phase 2 turns out to need a more
rigorous calibrated quantization scheme than bitsandbytes' calibration-free absmax
approach), `flash-attn` (a real latency win for Phi-3 but an extra build-from-source step;
add later if the report-only latency numbers need to look more realistic).

---

## 3. Models and datasets to download

### Model
**microsoft/Phi-3-mini-4k-instruct** (MIT license, no gated access) — this is the only
model weight download v1 needs (§3).

```bash
huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
  --local-dir ./models/phi-3-mini-4k-instruct
```

Equivalent Python (if preferred inside a setup script):
```python
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Phi-3-mini-4k-instruct", local_dir="./models/phi-3-mini-4k-instruct")
```

### Datasets (pulled via `datasets`, not `huggingface-cli`, but listed here since they're
fetched from the Hub the same way models are)
- **WikiText-2** (fitness-loop accuracy signal, §5):
  `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`
- **C4** (validation pass only, §5) — the full dataset is enormous; use streaming or a
  small fixed slice: `datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)`
- **PTB** (validation pass only, §5): `datasets.load_dataset("ptb_text_only", "penn_treebank")`
  — **flagged risk**: this dataset uses a loading script, and recent `datasets` versions
  have been dropping script-based dataset support; confirm it still loads in Phase 0, and
  have a fallback (e.g. a mirrored/parquet version of PTB, or substituting a different
  third perplexity set) ready if not.

---

## Open questions before running experiments (not blocking the plan, but worth deciding before Phase 6)

- **Stopping criterion**: with no discrete generations, what ends a run — a fixed number
  of evaluations, a wall-clock budget, or a convergence check (no improvement in the top-k
  for N evaluations)? Needs picking before Phase 6, doesn't affect Phases 0–5.
- **Coordination store past v1**: SQLite is proposed for the single-host case (Phase 4).
  §6 already flags the multi-host coordination mechanism as a design/build-phase decision
  — worth revisiting once the second GPU desktop actually gets added, since SQLite over a
  shared network path handles concurrent writes worse than a small dedicated service would.
