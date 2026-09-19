# Requirements: Evolutionary Search for Mixed-Precision LLM Quantization

This is the outline for the project. The core idea: use a genetic algorithm to search
per-layer mixed-precision quantization configurations for a small open-source transformer,
treating it explicitly as a modern application of the same evolutionary/meta-heuristic
optimization framework from your thesis work — that framing is what makes this project
valuable for d-Matrix specifically, not just another quantization repo.


## 1. Goal
Find a per-layer quantization configuration (which layers run at INT8, INT4, or FP16) that
minimizes model size and inference latency while keeping accuracy degradation under a
target threshold — using a genetic algorithm to search the configuration space, rather
than a fixed heuristic or uniform quantization.

### 2. Why this project, specifically
Most quantization repos apply a known recipe (e.g., uniform INT8, or a published
mixed-precision heuristic). This project instead treats quantization-config search as a
combinatorial optimization problem and applies population-based search to it — directly
continuous with the "asynchronous parallel steady-state genetic algorithm" and "dynamic
load balancing" work from my dissertation.

### 3. Model and scope
- **Model: Phi-3-mini-4k-instruct (3.8B params)**, MIT-licensed, modern architecture
  (GQA, SwiGLU, RoPE). Chosen specifically because it sits right at the edge of the
  target dev GPU's budget: on an RTX 3080 (10GB VRAM, ~9.5GB usable after CUDA/driver
  overhead), FP16 weights alone are ≈7.6GB, leaving only ~1.9GB for KV cache +
  activations + overhead — genuinely tight at 4K context. This makes mixed-precision
  quantization load-bearing for the project's story: it's what buys back headroom for a
  usable context length, not just a latency nice-to-have.
- Target dev hardware: Core i7-11700K (8 cores), RTX 3080 10GB VRAM, 64GB system RAM, as
  the primary/local host. All evaluation hosts are GPU-equipped for v1 — CPU-only hosts
  are out of scope (§6). A second desktop with its own GPU is available and kept open as
  an optional additional worker to add as the project progresses; v1 doesn't depend on it.
- Scope to weight quantization first (INT8/INT4 per linear layer or per transformer
  block), applied via **real quantization kernels** (bitsandbytes-style block-wise
  int8/int4 quantization) — not simulated/fake quantization. This is **not** the typical
  HF `load_in_4bit`/`load_in_8bit` flow, which quantizes an entire model once at load time
  into fixed custom tensor types and isn't designed to be reconfigured per layer on every
  GA evaluation. Instead, each layer's INT8/INT4 weights are precomputed once and swapped
  into a single live model per genome at evaluation time (§6). Both the accuracy forward
  pass and any wall-clock latency benchmarking run the actual quantized kernels on the
  assembled model. Activation quantization and KV-cache quantization are natural
  extensions, not required for v1.

---------------------------------------------

### 4. Search space and genome encoding
- Genome = one gene per quantizable unit, each gene taking a value from a small discrete
  set: {FP16, INT8, INT4} (extendable to include "skip quantization" for sensitive layers
  like embeddings/output head).
- **Granularity for v1**: Phi-3-mini has 32 transformer blocks. One gene per individual
  block gives a genome length of 32 and a search space of 3^32 ≈ 1.85×10^15 — far too
  large for a population of 20–50 to explore meaningfully without getting stuck in local
  optima. Group the 32 blocks into **8 contiguous super-blocks** (4 transformer blocks
  each, sharing one gene/precision) for v1 — genome length 8, search space 3^8 = 6,561,
  small enough for a population this size to meaningfully cover while still coarser than
  per-weight-tensor. Finer granularity (16 super-blocks, or eventually one gene per
  transformer block) is a natural v2 step once v1 demonstrates the method works at all
  (§9).

### 5. Fitness function
Multi-objective, combined into a scalarized fitness:
- **Accuracy term**: real quantized forward pass (per-layer precision from the genome,
  assembled from the precomputed weight bank, §6) evaluated as perplexity delta on
  WikiText-2 relative to the FP16 baseline. `accuracy_penalty = perplexity_delta /
  baseline_perplexity` — a fraction of the **fixed** FP16 baseline, not a rank within the
  current population.
- **Efficiency term**: computed **analytically** from the genome's per-layer bit-widths —
  `bytes(genome) = Σ param_count_i × bits_i / 8` — compared against the FP16 baseline's
  byte count. This is deliberately *not* measured wall-clock latency (see §6 for why).
  `efficiency_gain = 1 − bytes(genome) / bytes_fp16` — again a fraction of the **fixed**
  FP16 baseline, not a rank within the current population.
- Fitness = efficiency_gain × w1 − accuracy_penalty × w2, where **w1/w2 are fixed,
  tunable scalar hyperparameters** set once per run (e.g., default w1 = w2 = 0.5).
  **Correction from an earlier draft**: both terms were previously defined as percentile
  ranks within the live population. That's a real bug in a steady-state design — inserting
  any individual shifts the rank, and therefore the fitness, of every other individual
  currently in the population. That silently invalidates the fitness cache (a cached score
  goes stale the moment the population changes) and makes an individual's fitness depend
  on *when* it was evaluated rather than being an intrinsic property of its genome, which
  also undermines elitism. Normalizing each term against the fixed FP16 baseline instead
  gets the same scale-comparability (both terms land in a similar 0-to-~1 range) without
  any population-dependence: fitness is now a pure, deterministic function of the genome
  and the baseline constants — exactly what makes caching it sound.
- Real measured wall-clock latency (on GPU-equipped hosts only, using the same real
  quantization kernels) is still collected, but as a **report-only diagnostic metric** for
  the final Pareto plot (§8) — never fed into the GA's fitness score. See §6.
- **Evaluation set**: WikiText-2 perplexity, over a fixed token subset (§6), is the fitness
  signal used for every evaluation during the search itself, to keep the steady-state
  loop's per-step cost low. A broader validation pass — WikiText-2 + C4 + Lambada
  perplexity — runs once, after the search, on the baselines (§7) and the GA's final
  surviving population, to confirm the result isn't an artifact of overfitting to one
  dataset's quirks. This broader pass is cheap because it only runs on a handful of
  configs, not on every individual the search ever touches. (Lambada replaces PTB from an
  earlier draft — PTB's standard HF `datasets` loader depends on a deprecated Python
  loading script and no longer runs; see `Doc/implementation_plan.md` §3 for the
  verified replacement.)

### 6. GA mechanics
- **Architecture: asynchronous, parallel, steady-state** — continuous with the
  dissertation's "asynchronous parallel steady-state genetic algorithm with dynamic load
  balancing" (§2), not a generational design. There is one shared, fixed-capacity,
  fitness-sorted population; there are no generation boundaries or synchronization
  barriers between hosts.
- **Population**: fixed capacity of 20–50 individuals, maintained as a live sorted list
  (by fitness, §5).
- **Per-layer weight bank (prerequisite)**: standard bitsandbytes usage (e.g. HF's
  `load_in_4bit`/`load_in_8bit`) quantizes an entire model once at load time into custom
  tensor types — it isn't designed to be reconfigured per layer on every GA evaluation,
  and re-quantizing from scratch for each of potentially thousands of evaluations isn't
  viable. Instead: at startup, precompute each quantizable (super-)layer's INT8 and INT4
  weights **once** (calibration-free block-wise/absmax quantization), and keep all three
  precision variants of every layer's weights resident in **CPU RAM** (64GB available —
  comfortably fits 3 copies of a 3.8B-param model, ≈13GB total). A single live model
  instance is kept on GPU; evaluating a genome means copying only the layers whose
  precision that genome needs into that instance (most children differ from their parent
  by only a few genes, so most layers don't need re-copying) before the forward pass. This
  avoids ever holding three full-precision copies of the whole model in VRAM at once
  (≈13GB — over the RTX 3080's 10GB budget) while keeping per-evaluation overhead to a
  fast host-to-device copy rather than a re-quantization.
- **Worker step** — run independently and continuously by every GPU-equipped worker host
  whenever it has spare compute, with no coordination needed between hosts beyond the
  shared population state:
  1. Select two parents from the live population via **linear rank-weighted selection**:
     `P(rank i) = (2 − s)/N + 2i(s − 1) / (N(N − 1))`, the standard Baker linear-ranking
     formula, where `i` is the individual's rank (**0 = worst, N−1 = best** — rank
     increases with fitness) among the live population of size `N`, and `s` is a fixed
     **selection pressure** hyperparameter, `1.0 ≤ s ≤ 2.0` (default `s = 1.5` for v1).
     **Correction from an earlier draft**: this section previously said "0 = best,
     N−1 = worst", backwards from what the formula actually computes (`P(0) = (2−s)/N`
     is the *smaller* value for `s > 1`, `P(N−1) = s/N` the larger) — caught by a Phase 4
     unit test asserting the wrong direction against the real implementation. `s = 1.0`
     gives exactly uniform random selection (no pressure, well-defined, not just a
     limiting case); `s = 2.0` gives the sharpest linear falloff, with the single worst
     individual reaching zero selection probability. Chosen over strict 1/rank weighting
     for explicitly tunable, gentler pressure, reducing the risk of collapsing diversity
     in a population this small (20–50).
  2. Produce one child via crossover (single or multi-point, on the per-layer gene array)
     of the two parents, then apply mutation (per-gene random reassignment to a different
     precision level, tunable mutation rate).
  3. Hash the child's genome and check it against the shared fitness cache; on a hit,
     reuse the cached result and skip evaluation.
  4. On a cache miss, assemble the genome from the weight bank (above), evaluate the
     child's fitness (§5), and store the result in the cache keyed by genome hash.
  5. Insert the child into the live sorted population; if this exceeds capacity, drop the
     current worst-ranked individual.
- **Fitness cache**: a genome-hash → fitness lookup shared across all workers, storing
  only scalar metadata (fitness, perplexity delta, byte size) — never model weights or
  tensor references. It covers every genome evaluated anywhere in the run, not just those
  currently in the live population. This is only sound because §5 now defines fitness as
  a pure function of the genome and the fixed FP16 baseline; under the earlier
  population-percentile-rank definition, a cached score would have silently gone stale
  every time the population changed. Mutation reverting a gene, crossover reconstructing
  an already-seen individual, and reproduced near-duplicates late in the search all hit
  this cache in practice, saving real compute.
- **Elitism**: emerges by construction rather than as a special rule — an individual only
  leaves the population when a strictly better one is inserted and the population is at
  capacity, so the current best individuals are never evicted.
- **Coordination requirement**: this design needs a shared, concurrently-mutable
  population + fitness-cache store that every worker host reads from and writes to (e.g.
  a lightweight coordinator process or a lock-protected shared store), replacing the
  "each host owns an independent generation slice" model a generational design would
  need. The specific mechanism (in-process, a small service, a shared DB) is a
  design/build-phase decision, not a requirements-level one.
- **Hosts**: v1 uses GPU-equipped hosts only — CPU-only evaluation isn't viable at this
  model size (perplexity eval on a 3.8B-param model without GPU acceleration would take
  minutes-to-hours per individual, so a CPU host would contribute near-zero throughput to
  the steady-state queue relative to a GPU host, making the added complexity not worth
  it). v1 runs on the local RTX 3080 host; a second desktop with its own GPU is an option
  to add later as an additional worker — the architecture below already supports this
  with no changes needed.
- **Heterogeneous hardware**: because there's no generation barrier, a faster or
  less-busy GPU host naturally completes more worker steps per unit time than a slower
  one — load balancing falls out of the architecture rather than needing explicit logic.
  Mixing measured wall-clock latency from different hardware into fitness would still
  confound the search (a config evaluated on a slower host would rank worse for reasons
  unrelated to its quantization), which is why the efficiency term stays the analytical
  byte-size proxy (§5) regardless of which host evaluates it. Each evaluating host
  additionally benchmarks real wall-clock latency (real quantization kernels) for the
  individuals it evaluates, purely as report-only data for the final plot (§8) — it never
  feeds back into selection. **Determinism note**: different GPU models or driver
  versions can produce slightly different floating-point results for the same genome,
  which — if two hosts independently evaluate the same genome — could in principle write
  conflicting values into the fitness cache. Mitigated by a fixed random seed and the
  fixed WikiText-2 token subset (§5), and treated as a documented limitation rather than
  something v1 needs to fully eliminate — quantization's effect on perplexity is far
  larger than cross-hardware floating-point noise.

### 7. Baselines to compare against
- FP16 baseline (upper bound on accuracy, worst on efficiency).
- Uniform INT8 and uniform INT4 (naive baselines).
- A simple heuristic baseline, e.g., "quantize middle layers more aggressively than first/last layers" (common published heuristic) — this is what the GA needs to beat to make the project's point.

### 8. Deliverables
- Code: GA implementation + fitness evaluation harness + baseline comparisons, clean enough to read (this audience will look at code quality and math clarity, not just results).
- Results plots: accuracy vs. efficiency Pareto front, GA-found configs vs. baselines —
  shown against (a) the analytical byte-size metric the GA actually optimized against, and
  (b) real measured wall-clock latency from GPU hosts, to show the analytical proxy tracks
  something real.
- A short write-up (README or blog post) explicitly framing this as "applying
  population-based combinatorial optimization to a modern LLM inference efficiency
  problem." This framing is the actual point of the project.

### 9. Time-boxing
- V1 (baseline-worthy): coarse per-block genome, scalarized fitness, small model, 1–2 weeks of part-time effort.
- V2 (stretch, if time allows): Pareto-front NSGA-II version, finer-grained genome, maybe extend to activation quantization. This is what would separate the project from a weekend demo.
