# Evol_inference

**Evolutionary search for mixed-precision LLM quantization** - applying population-based
combinatorial optimization to a modern LLM inference efficiency problem, directly
continuous with the asynchronous parallel steady-state genetic algorithm and dynamic
load balancing work from my dissertation.

Most quantization repos apply a known recipe (uniform INT8, or a published
mixed-precision heuristic). This one treats the choice of per-layer precision as a
combinatorial search problem and lets a steady-state GA search it, rather than assuming
the heuristic is right. Full design rationale, including two rounds of external review
and the corrections that came out of them, is in [`Doc/requirements.md`](Doc/requirements.md);
the phase-by-phase build log is in [`Doc/implementation_plan.md`](Doc/implementation_plan.md).

## The approach

- **Model**: [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  (3.8B params), chosen because its FP16 weights (≈7.6GB) sit right at the edge of a
  10GB consumer GPU's budget - quantization isn't a nice-to-have here, it's what buys
  back headroom for real context length.
- **Genome**: 8 genes, one per super-block of 4 consecutive transformer layers
  (grouped from the model's 32 layers to keep the search space tractable -
  3<sup>8</sup> = 6,561 configurations, not 3<sup>32</sup>). Each gene is one of
  `{FP16, INT8, INT4}`, applied via real bitsandbytes kernels through module
  substitution - not simulated quantization, and not the standard "quantize the whole
  model once at load time" flow, since neither supports reconfiguring per layer on
  every GA evaluation.
- **Fitness**: `efficiency_gain × w1 − accuracy_penalty × w2`, where both terms are
  fractions of a fixed FP16 baseline (not a percentile rank within the population - an
  earlier draft had that bug, and a steady-state design can't tolerate it: a
  population-relative fitness silently invalidates every cached score the moment the
  population changes). `accuracy_penalty` is WikiText-2 perplexity delta on a fixed
  2048-token slice; `efficiency_gain` is the analytical byte-size reduction from the
  genome's bit-widths.
- **Search**: an asynchronous, parallel, steady-state GA - no generations, no
  synchronization barrier. One fixed-capacity population (40 individuals) that workers
  continuously pull parents from (linear rank-weighted selection) and insert children
  into (crossover + mutation), evicting the current worst on overflow. Elitism falls
  out of that eviction rule rather than needing a special case.

## Results

Four baselines (§7) and the GA's best-found configuration, scored on the same
fitness-loop signal:

| config                               | fitness    | accuracy_penalty | efficiency_gain | perplexity | size (GB) | latency (ms/fwd) |
|--------------------------------------|------------|------------------|-----------------|------------|-----------|------------------|
| FP16                                 | 0.0000     | 0.0000           | 0.0000          | 4.3708     | 6.750     | 383.9            |
| uniform INT8                         | 0.2423     | 0.0154           | 0.5000          | 4.4381     | 3.375     | 309.1            |
| heuristic (aggressive middle layers) | 0.2106     | 0.0164           | 0.4375          | 4.4424     | 3.797     | 346.3            |
| **uniform INT4 (best baseline)**     | **0.3448** | 0.0604           | 0.7500          | 4.6346     | 1.688     | 397.0            |
| **GA best** (126 evaluations)        | 0.3349     | 0.0490           | 0.7188          | 4.5851     | 1.898     | 385.2            |

The GA converged on `[INT4, INT4, INT4, INT4, INT4, INT8, INT4, INT4]` - uniform INT4
with exactly one super-block bumped up to INT8 - trading a little efficiency for
meaningfully better accuracy than pure uniform INT4. It came close to the best
baseline's scalarized fitness (within 3%) but didn't beat it outright within its
evaluation budget.

That single scalarized number undersells what the search actually found, though:

![Accuracy vs. efficiency, analytical byte-size metric](results/pareto_efficiency.png)

Several GA-found individuals land directly **on** the true Pareto front between
uniform INT8 and uniform INT4 - these are tradeoff points that neither naive baseline
occupies. The heuristic baseline ("quantize the middle layers more aggressively"), which
§7 frames as what the GA needs to beat, is actually dominated by uniform INT8 here:
keeping any super-blocks at full FP16 costs more in efficiency than the INT4 blocks save.

The second plot is the more important one, and the result is less flattering to
the analytical proxy:

![Accuracy vs. efficiency, real measured latency](results/pareto_latency.png)

**Real measured wall-clock latency does not track the analytical byte-size metric.**
FP16 and uniform INT8 dominate the *measured-latency* Pareto front; uniform INT4 -
the smallest configuration by far - is actually the **slowest** in real forward-pass
latency. This is a known characteristic of bitsandbytes' 4-bit dequantize-and-matmul
path at small batch sizes: the per-block dequantization overhead can outweigh the
memory-bandwidth savings that make INT4 a win at larger scale. The GA optimized
correctly against the metric it was given; that metric just isn't the whole story for
real deployment latency on this hardware. Both plots are generated from the same run
data - see `scripts/phase7_report.py`.

**Validation across three independent datasets** (WikiText-2, C4, Lambada - Phase 7,
run once on baselines + the GA's final population, not on every individual the search
touched) confirms the fitness-loop signal generalizes: the same accuracy ordering
(FP16 < uniform INT8 < heuristic < uniform INT4) holds on all three datasets, not just
the one the search optimized against.

| config       | WikiText-2 | C4     | Lambada |
|--------------|------------|--------|---------|
| FP16         | 4.3708     | 9.3108 | 15.0458 |
| uniform INT8 | 4.4381     | 9.4063 | 15.2645 |
| heuristic    | 4.4424     | 9.4966 | 15.3303 |
| uniform INT4 | 4.6346     | 9.7555 | 15.9620 |

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/hf download microsoft/Phi-3-mini-4k-instruct --local-dir ./models/phi-3-mini-4k-instruct

PYTHONPATH=. .venv/bin/python -m pytest tests/                    # 55 tests

PYTHONPATH=. .venv/bin/python scripts/phase5_baselines.py         # baselines table
PYTHONPATH=. .venv/bin/python scripts/phase6_run_search.py        # runs the GA search
PYTHONPATH=. .venv/bin/python scripts/phase7_report.py            # validation + plots
```

## Layout

- `evol_inference/` - the library: `weight_bank.py` (per-layer quantized weight bank +
  genome assembly), `fitness.py` (accuracy/efficiency scoring + caching), `ga.py`
  (steady-state population, selection, crossover, mutation), `search.py` (the
  evaluation-budget / stagnation stopping criterion), `baselines.py`, `pareto.py`,
  `validation.py`.
- `scripts/` - one runnable entry point per build phase.
- `tests/` - pytest suite; logic that doesn't touch the model (selection math,
  crossover, Pareto fronts, the stopping criterion) is tested without a GPU dependency.
- `Doc/` - requirements, implementation plan, and the review history that shaped both.
