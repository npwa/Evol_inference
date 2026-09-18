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
- Target dev hardware: Core i7-11700K (8 cores), RTX 3080 10GB VRAM, 64GB system RAM.
- Scope to weight quantization first (INT8/INT4 per linear layer or per transformer
  block), applied via **real quantization kernels** (bitsandbytes/GPTQ-style) — not
  simulated/fake quantization. Both the accuracy forward pass and any wall-clock latency
  benchmarking run the actual quantized kernels. Activation quantization and KV-cache
  quantization are natural extensions, not required for v1.

---------------------------------------------

### 4. Search space and genome encoding
- Genome = one gene per quantizable unit (per-layer or per-block), each gene taking a
  value from a small discrete set: {FP16, INT8, INT4} (extendable to include "skip
  quantization" for sensitive layers like embeddings/output head).
- For a model with N transformer blocks, genome length ≈ N (or N × number of quantizable
  sub-modules per block, if going finer-grained).
- Keep the search space per-block rather than per-weight-tensor for v1 — coarser
  granularity, faster convergence, still demonstrates the method.

### 5. Fitness function
Multi-objective, combined into a scalarized fitness:
- **Accuracy term**: real quantized forward pass (per-layer precision from the genome,
  applied via bitsandbytes/GPTQ-style kernels) evaluated as perplexity delta on WikiText-2
  relative to the FP16 baseline. `accuracy_penalty` = the individual's percentile rank
  (0–1) of perplexity delta *within the live population at the time of evaluation* (worse
  accuracy → rank closer to 1). There are no generation boundaries (§6 is a steady-state
  design), so this rank is computed against whatever the population currently holds at
  insertion time, not a fixed generation snapshot.
- **Efficiency term**: computed **analytically** from the genome's per-layer bit-widths —
  `bytes(genome) = Σ param_count_i × bits_i / 8` — compared against the FP16 baseline's
  byte count. This is deliberately *not* measured wall-clock latency (see §6 for why).
  `efficiency_gain` = the individual's percentile rank (0–1) of analytical byte-size
  reduction *within the live population at the time of evaluation* (larger reduction →
  rank closer to 1).
- Fitness = efficiency_gain × w1 − accuracy_penalty × w2, where **w1/w2 are fixed,
  tunable scalar hyperparameters** set once per run (e.g., default w1 = w2 = 0.5) — *not*
  themselves derived from population statistics. The percentile-rank normalization above
  is what makes the two raw terms (a perplexity number, a byte count) comparable on the
  same 0–1 scale before the fixed weights combine them.
- Real measured wall-clock latency (on GPU-equipped hosts only, using the same real
  quantization kernels) is still collected, but as a **report-only diagnostic metric** for
  the final Pareto plot (§8) — never fed into the GA's fitness score. See §6.
- **Evaluation set**: WikiText-2 perplexity is the fitness signal used for every
  evaluation during the search itself (fast, keeps the steady-state loop's per-step cost
  low even on CPU-only hosts). A broader validation pass — WikiText-2 + C4 + PTB
  perplexity — runs once, after the search, on the baselines (§7) and the GA's final
  surviving population, to confirm the result isn't an artifact of overfitting to one
  dataset's quirks. This broader pass is cheap because it only runs on a handful of
  configs, not on every individual the search ever touches.

### 6. GA mechanics
- **Architecture: asynchronous, parallel, steady-state** — continuous with the
  dissertation's "asynchronous parallel steady-state genetic algorithm with dynamic load
  balancing" (§2), not a generational design. There is one shared, fixed-capacity,
  fitness-sorted population; there are no generation boundaries or synchronization
  barriers between hosts.
- **Population**: fixed capacity of 20–50 individuals, maintained as a live sorted list
  (by fitness, §5).
- **Worker step** — run independently and continuously by every host (GPU or CPU-only)
  whenever it has spare compute, with no coordination needed between hosts beyond the
  shared population state:
  1. Select two parents from the live population via **linear rank-weighted selection**
     — selection probability decreases linearly with fitness rank (best individual most
     likely, worst least likely, linear falloff in between). Chosen over strict 1/rank
     weighting because it gives gentler selection pressure, reducing the risk of
     collapsing diversity in a population this small (20–50).
  2. Produce one child via crossover (single or multi-point, on the per-layer gene array)
     of the two parents, then apply mutation (per-gene random reassignment to a different
     precision level, tunable mutation rate).
  3. Hash the child's genome and check it against the shared fitness cache; on a hit,
     reuse the cached result and skip evaluation.
  4. On a cache miss, evaluate the child's fitness (§5) and store the result in the cache
     keyed by genome hash.
  5. Insert the child into the live sorted population; if this exceeds capacity, drop the
     current worst-ranked individual.
- **Fitness cache**: a genome-hash → fitness lookup shared across all workers, covering
  every genome evaluated anywhere in the run (not just those currently in the live
  population). Mutation reverting a gene, crossover reconstructing an already-seen
  individual, and reproduced near-duplicates late in the search all hit this cache in
  practice, saving real compute.
- **Elitism**: emerges by construction rather than as a special rule — an individual only
  leaves the population when a strictly better one is inserted and the population is at
  capacity, so the current best individuals are never evicted.
- **Coordination requirement**: this design needs a shared, concurrently-mutable
  population + fitness-cache store that every worker host reads from and writes to (e.g.
  a lightweight coordinator process or a lock-protected shared store), replacing the
  "each host owns an independent generation slice" model a generational design would
  need. The specific mechanism (in-process, a small service, a shared DB) is a
  design/build-phase decision, not a requirements-level one.
- **Heterogeneous hardware**: because there's no generation barrier, fast (GPU) hosts
  naturally complete more worker steps per unit time than slow CPU-only hosts — load
  balancing falls out of the architecture rather than needing explicit logic. Mixing
  measured wall-clock latency from different hardware into fitness would still confound
  the search (a config evaluated on a slow host would rank worse for reasons unrelated to
  its quantization), which is why the efficiency term stays the analytical byte-size
  proxy (§5) for every host regardless of speed. GPU-equipped hosts additionally run a
  real wall-clock latency benchmark (real quantization kernels) for the individuals they
  evaluate, purely as report-only data for the final plot (§8) — it never feeds back into
  selection.

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
