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
- We'll use a small, well-understood open model to keep iteration cycles fast: something
  in the 125M–1.3B range (e.g., GPT-2 small/medium, Pythia-410M, or TinyLlama). Small
  enough to run many search generations on a single GPU; large enough to have a real
  per-layer accuracy/latency tradeoff to exploit.
- Scope to weight quantization first (INT8/INT4 per linear layer or per transformer
  block). Activation quantization and KV-cache quantization are natural extensions, not
  required for v1.

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
- **Accuracy term**: perplexity delta on a held-out set (WikiText-2 or similar), or task
  accuracy on a small benchmark (e.g., a subset of lm-eval-harness tasks) relative to the
  FP16 baseline.
- **Efficiency term**: measured or estimated inference latency and/or model size (bytes)
  for the given config.
- Find a way to define `accuracy_penalty` as a value indicating the relative loss of
  accuracy across the testing set. and define the `efficiency_gain` the relative
  improvement of inference performance.
- Fitness = efficiency_gain × w1 - accuracy_penalty × w2, with w1/w2 defined as normalized
  rank across the population.

### 6. GA mechanics
- **Population**: 20–50 candidate configs per generation (small population is fine given fast fitness eval on a small model).
- **Selection**: tournament selection.
- **Crossover**: single or multi-point crossover on the per-layer gene array.
- **Mutation**: per-gene random reassignment to a different precision level, with mutation rate as a tunable parameter — directly connects to the "self-organizing control architecture"
- **Elitism**: carry top-k configs forward unchanged each generation to guarantee monotonic improvement.
- **Parallelization**: evaluate the population's fitness in parallel across available
  GPU/CPU resources. We have a number of resources, some CPU/GPU and some CPU
  only. Therefore some hosts will be able to accurately (albeit slowly) measure accuracy
  (but not with a fair efficiency measurement). does this make sense? can we abandon
  efficiency measurement on CPU only hosts for individual combinations if accuracy falls
  below 50% rank?

### 7. Baselines to compare against
- FP16 baseline (upper bound on accuracy, worst on efficiency).
- Uniform INT8 and uniform INT4 (naive baselines).
- A simple heuristic baseline, e.g., "quantize middle layers more aggressively than first/last layers" (common published heuristic) — this is what the GA needs to beat to make the project's point.

### 8. Deliverables
- Code: GA implementation + fitness evaluation harness + baseline comparisons, clean enough to read (this audience will look at code quality and math clarity, not just results).
- A results plot: accuracy vs. efficiency Pareto front, GA-found configs vs. baselines.
- A short write-up (README or blog post) explicitly framing this as "applying
  population-based combinatorial optimization to a modern LLM inference efficiency
  problem." This framing is the actual point of the project.

### 9. Time-boxing
- V1 (baseline-worthy): coarse per-block genome, scalarized fitness, small model, 1–2 weeks of part-time effort.
- V2 (stretch, if time allows): Pareto-front NSGA-II version, finer-grained genome, maybe extend to activation quantization. This is what would separate the project from a weekend demo.
