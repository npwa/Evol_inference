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
**Revised** after a second review caught a real problem with the original plan here:
raw `bitsandbytes.functional` calls produce packed low-bit tensors that carry required
metadata (`QuantState` — absmax tables, block size, double-quant offsets) that a plain
`nn.Linear.forward` (`F.linear(input, weight, bias)`) doesn't know how to interpret. Two
ways that breaks: either it errors outright on the dtype/shape mismatch, or — worse —
if forced to work by dequantizing back to FP16 before the matmul, it silently stops
running real low-bit compute. That second failure mode wouldn't be caught by an accuracy
check (the numeric output can come out right, since full dequant-then-multiply is
mathematically consistent with the same quantization scheme) — it would only show up
later as the report-only latency benchmark (§5/§6) showing no speedup for INT4 vs FP16,
quietly defeating the entire reason real kernels were chosen over simulated quantization
in the first place (§3).
1. Use `bitsandbytes.nn.Linear8bitLt` and `bitsandbytes.nn.Linear4bit` — the module
   wrappers that package the quantized weight together with its `QuantState` and dispatch
   into the real low-bit GEMM path — rather than raw `bitsandbytes.functional` calls
   against generic linear layers.
2. For each of the 8 super-blocks' quantizable linear layers, precompute and instantiate
   all three variants once (FP16 `nn.Linear`, `Linear8bitLt`, `Linear4bit`), keeping the
   currently-unused variants' quantized weights on **CPU** (all three variants for the
   whole model total ≈13GB, over the 10GB VRAM budget if all resident on GPU at once).
3. Implement "assemble a genome" as **module substitution**, not tensor mutation: for
   each super-block, move the genome-selected precision variant's module to
   `cuda` (`.to("cuda")`) and reassign the parent container's attribute to point at it
   (e.g. `block.mlp.gate_up_proj = precomputed_variant`); move the super-block's
   previously-active variant back to CPU to free VRAM. Most children differ from their
   parent by only a few genes, so most super-blocks don't need to move on a given
   evaluation — full-model-sized transfers are the worst case, not the typical case.
4. Verify correctness *and* that real kernels are actually running: an all-INT8 genome's
   output should be close to bitsandbytes' standard `load_in_8bit` output on the same
   input; an all-FP16 genome should exactly reproduce the FP16 baseline; and — the check
   that would have caught the original plan's silent-failure mode — an all-INT4 genome's
   forward pass should show measurably lower VRAM usage and different latency than the
   all-FP16 genome's. If it doesn't, the swap is falling back to a dequant-heavy path
   instead of dispatching into the real low-bit kernel.
5. **Performance note (not a v1 blocker)**: repeated `.to("cuda")`/`.to("cpu")` moves
   could in principle fragment PyTorch's CUDA caching allocator over many evaluations. Not
   expected to matter at v1's evaluation budget (Phase 6) given most swaps only touch 1-2
   of 8 super-blocks, but if profiling later shows OOM or slowdown, the fallback is
   pre-allocating fixed-size GPU buffers per super-block and using in-place `copy_()`
   instead of `.to()`/module reassignment.

### Phase 3 — Fitness harness (§5)
1. WikiText-2 perplexity evaluation over a **fixed token subset** (needed for the
   determinism note in §6) — implement once, reuse for every genome.
2. Analytical `bytes(genome)` calculation from the genome's per-layer bit-widths.
3. `accuracy_penalty` / `efficiency_gain` / scalarized `fitness` per the fixed-baseline
   formulas in §5.
4. Genome-hash → fitness cache — an in-memory Python dict for v1 (see Phase 4 step 4 for
   why, and the persistence note there), scalar metadata only, per §6.
5. Real wall-clock latency benchmark (report-only metric, §6) — measured on GPU, logged
   alongside but never fed into fitness.

### Phase 4 — GA engine (§6)
1. Genome representation: 8-gene array over `{FP16, INT8, INT4}`.
2. Linear rank-weighted parent selection (`s = 1.5` default).
3. Crossover (single/multi-point) + mutation (tunable per-gene reassignment rate).
4. Steady-state population as a fixed-capacity sorted list (worst evicted on overflow),
   kept **in-memory** (plain Python list/dict) for v1, not SQLite — a second review
   pointed out that a single-host, effectively-sequential worker loop (step 5 below)
   gains nothing from a DB round-trip on every evaluation over a plain in-process
   structure. Periodically snapshot both the population and the fitness cache to a
   JSON/pickle file (e.g. every N insertions) so a crash doesn't lose an unattended run.
   Revisit this once a second host joins (see the coordination-store note below) — SQLite
   or in-memory-only stops being an option once more than one process needs to write to
   the same state.
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
1. Run the steady-state loop until either: **(a)** 250–300 total evaluations complete
   (with a search space of 3^8 = 6,561 genomes after §4's 8-super-block grouping, 300
   evaluations samples ~4.5% of the space — enough for a population-based search to make
   real progress without an open-ended budget), or **(b)** the global best fitness hasn't
   improved in 30 consecutive evaluations (stagnation), whichever comes first.
2. Snapshot the final population + baselines.

### Phase 7 — Validation + deliverables (§5, §8)
1. Run the broader WikiText-2 + C4 + Lambada perplexity validation pass on baselines +
   the GA's final population only (cheap — a handful of configs, §5).
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
- `datasets` (WikiText-2, C4, Lambada — all pulled from the HF Hub)
- `huggingface_hub` (model download CLI/API)

**GA / coordination**
- nothing beyond the standard library for v1 (`hashlib`, `random`, `dataclasses`, `json`/
  `pickle` for periodic snapshots — all stdlib). Population + fitness cache are in-memory
  Python structures for v1 (Phase 4); see the coordination-store note below for what
  replaces this once a second host joins.

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
- **Lambada** (validation pass only, §5), replacing PTB from the original plan:
  `datasets.load_dataset("EleutherAI/lambada_openai", "en")`. PTB's standard HF loader
  (`ptb_text_only`) was checked and confirmed broken — it depends on a Python loading
  script, and current `datasets` versions refuse to run it ("Dataset scripts are no
  longer supported"), which would have failed Phase 7 outright. `EleutherAI/lambada_openai`
  is confirmed parquet-backed (no script dependency) and, as a narrative-text word-
  prediction benchmark, is a better fit than PTB or WikiText-103 anyway for what this
  third dataset is actually for (§5): checking the result generalizes beyond one
  dataset's quirks, which benefits from a genuinely different domain/style — WikiText-103
  would mostly just be "more Wikipedia," largely the same style as WikiText-2.

---

## Resolved from the previous draft's open questions

- **Stopping criterion**: resolved — see Phase 6 (250–300 evaluations or 30-evaluation
  stagnation, whichever comes first).
- **Coordination store past v1**: v1 uses an in-memory population + fitness cache
  (Phase 4), not SQLite — simpler and sufficient for a single, effectively-sequential
  worker. When the second GPU desktop actually joins, don't reach for SQLite over a
  shared/network path either — it's prone to `database is locked` errors under concurrent
  writes from multiple hosts. A small dedicated coordinator (e.g. a lightweight FastAPI
  service, or a Redis instance on the primary host) is the better fit at that point. Still
  a design/build-phase decision (§6) — noted here so it isn't re-litigated from scratch
  later.
