# Implementation Plan: Arm Port (branch `mac-m4`) — Accuracy / Speed / Power Co-Optimization

Planning only — no code yet. Extends the base project (`Doc/implementation_plan.md`,
Phases 0–7) and builds on the energy harness described in `README_arm.md`
(`energy_meter.py`). Section references (§N) below are to this document unless stated.

## 0. Goal and framing

Re-target the mixed-precision search from "bitsandbytes on an RTX 3080" to "quantized
LLM inference on an Arm CPU (Apple M4, cloud bare-metal Mac)", and replace the
two-term fitness (accuracy vs. latency) with a **three-objective** one: **accuracy,
speed, and energy**. The portfolio story for the Arm Principal AI Performance Engineer
role is the same lesson as the README's primary finding, one level up: *the cost model
must be measured on the target silicon, not assumed.* On the 3080 the byte-size proxy
pointed the wrong way for latency; on Arm the interesting question is whether
latency-optimal and energy-optimal configurations coincide (they often don't — race-to-idle
vs. P-core/E-core power, memory-bound decode vs. compute-bound prefill).

### Job-description coverage (what each requirement maps to)

| Job requirement / keyword | Where this plan addresses it |
|---|---|
| Kernel-level optimization, "Triton, CUDA or other kernel level programming language" | §6 Track B: hand-written NEON/SDOT/I8MM int8 GEMV micro-kernel (C++ intrinsics), benchmarked against KleidiAI's; optional CUDA/Triton reference on the 3080 for the same op |
| Arm's AI optimization tools | KleidiAI (via llama.cpp `GGML_CPU_KLEIDIAI`), optional ExecuTorch + XNNPACK/KleidiAI backend (§8), Arm Performance Studio / Streamline / `perf` on Graviton (§7) |
| Arm ISA features | NEON, dotprod, **I8MM**, BF16, **SVE/SVE2** (Graviton), **SME/SME2** (M4) — feature detection recorded with every result (§3.1), kernel path attribution (§7) |
| Parallel computing, memory hierarchies, DNN perf techniques | Roofline analysis of prefill (compute-bound) vs. decode (bandwidth-bound) on M4 (§7); thread-count / P-vs-E-core study (§5.3) |
| Python and C++, modern AI frameworks, profiling tools | Python harness + C++ kernel track + llama.cpp / ExecuTorch; `powermetrics`, Instruments, `perf` (§7) |
| Power and performance (the role says both, repeatedly) | Energy as a first-class objective: J/token, tokens/s/W, EDP (§4) |
| Customer-facing / reference implementations / communication | Reproducible reference harness, "Arm porting guide" write-up, results deck (§10) |
| Edge-device AI optimization | M4 is the edge-class target; model ≤ 8B; everything measured at batch 1 (the edge case) |

## 1. Key decisions and constraints

1. **bitsandbytes does not run on Arm CPU/macOS.** Nothing in `weight_bank.py` ports.
   The quantization backend becomes **GGUF + llama.cpp** (CPU backend, KleidiAI enabled).
   This is the single largest piece of work — comparable to the compiled-runtime
   benchmark plan's "second parallel evaluation backend" (`Doc/implementation_plan_benchmark.md` §0).
2. **The genome stays the same shape** (8 super-blocks × a small precision alphabet), so
   the GA (`ga.py`, `search.py`), the genome-hash cache, and the README's findings carry
   over unchanged. Alphabet becomes `{F16, Q8_0, Q4_0}` initially; Q4_K variants are an
   optional 4th symbol (§5.2).
3. **Only measurements taken on the Arm target feed fitness for latency/energy.**
   (`README_arm.md`: Graviton SVE and M4 SME take different kernel paths.) Linux/x86 runs
   validate *correctness, plumbing and accuracy*, never Arm performance claims.
4. **Mac time is the scarce resource** (~$40/day, 24 h minimum). Design for: a
   one-command bootstrap, pre-staged artifacts, unattended queued runs, crash-safe
   snapshots, and *zero* debugging on the Mac that could have been done on Linux.
5. **Don't break `main`.** The 63 existing tests and the README results must remain
   reproducible. The port adds backends rather than replacing the bitsandbytes path; the
   GPU path is kept as the **accuracy cross-check and sensitivity reference** (§2.3).
6. **Python harness, not bash** (`README_arm.md` OS-compatibility rules): no
   `timeout`/`taskset`/GNU `sed`; thread count controlled by flag only on macOS.
7. Keep `Doc/requirements.md` conventions: fixed seeds, fixed eval slice, fixed-baseline
   normalization (fitness must remain a pure function of the genome *except* the noisy
   measured terms, which are handled explicitly in §4.3).

## 2. Development strategy: what runs where

### 2.1 Environment tiers (cheapest first; promote only when the tier below is green)

| Tier | Machine | What it proves | Cost |
|---|---|---|---|
| T0 | Local desktop, x86 Linux, no GPU | Unit tests, GA loop, GGUF assembler, parsers, mock/replay/RAPL meters | free |
| T1 | Local desktop + **RTX 3080** | Real-model end-to-end: CUDA-build llama.cpp for fast accuracy evals; bnb reference path; full GA run against *simulated* Arm latency/energy (§2.2) | free |
| T2 | Local desktop, **aarch64 emulation** (QEMU user-mode / `docker --platform linux/arm64`), small proxy model | The Arm build actually compiles and the NEON/dotprod/I8MM/KleidiAI code paths *execute correctly* (output matches x86 within tolerance). Correctness only; timings meaningless | free |
| T3 | AWS Graviton (c7g/c8g, ~$0.1–1/h, on-demand or spot) | Real Arm hardware, native SVE/I8MM, real `perf` counters, Linux (same OS family as T0 — easy to debug). Validates Arm *Linux* path and profiling workflow; **not** fitness numbers for M4 | cents–dollars |
| T4 | AWS `mac2-m4` bare metal (24 h block) | The only source of M4 latency/energy used in the final results | ~$40/day |

Rule: a bug found at T4 that T0–T3 could have caught is a process failure; the Mac
checklist (§9) is designed around that.

### 2.2 What the 3080 desktop is for (concretely)

1. **Fast accuracy evaluation during development.** A CUDA build of llama.cpp
   (`-DGGML_CUDA=ON`, `-ngl 99`) runs `llama-perplexity` over the fixed 2048-token slice
   in seconds, so the GGUF assembler and the whole GA loop can be exercised end-to-end
   on real Phi-3 weights with hundreds of evaluations in minutes — same loop the Mac
   will run, only the latency/energy probes are simulated.
2. **Sensitivity pre-computation (saves Mac hours).** Per-super-block, per-precision
   perplexity deltas (8 blocks × 2 precisions ≈ 16 evaluations + baseline) computed on
   the 3080 give the per-block accuracy table `a_i(b)` from the README's separability
   analysis. Uses: (a) seed part of the Mac GA's initial population with
   sensitivity-guided genomes; (b) an additive *surrogate* for accuracy to sanity-check
   the real values; (c) the README's "edge blocks are 30× more sensitive" finding
   becomes a cross-backend replication (bnb INT8 vs. GGUF Q8_0) — a result in its own right.
3. **Reference oracle**: the existing bnb path (`WeightBank`) provides accuracy numbers
   for the same genomes to quantify bnb-vs-GGUF quantization differences.
4. **RAPL as a real (x86) meter** to test baseline subtraction, noise handling, and
   repeat-measurement statistics against genuinely noisy power data (`/sys/class/powercap/intel-rapl`
   is present on this desktop; may need `chmod`/udev rule for non-root reads —
   verify).
5. **Simulated Arm cost model** ("`SimulatedArmProbe`"): latency/energy generated from a
   documented analytical model (bytes moved for decode, FLOPs for prefill, per-type
   dequant overhead constants, plus noise) so the three-objective GA machinery,
   Pareto ranking, plots and reports are all developed and tested on Linux. It is
   clearly labelled synthetic and can never be selected in a "real" run (§4.3 guard).
6. **Optional kernel track on the 3080** (Triton/CUDA int8 GEMV mirroring the Arm
   micro-kernel) — directly evidences the "Triton, CUDA" requirement (§6).

### 2.3 Cross-backend consistency checks (all runnable on T1)

- FP16 genome in llama.cpp vs. HF FP16: perplexity within ~0.5% (tokenization/slice
  alignment is the usual culprit — assert identical token IDs).
- Uniform Q8_0 vs. bnb INT8 and uniform Q4_0 vs. bnb INT4: same ordering
  (FP16 < INT8 < INT4); report the magnitude gap.
- CPU-build vs. CUDA-build llama.cpp perplexity on the same GGUF: tolerance-bounded
  (documents how much hardware-dependent kernel numerics move accuracy, which matters
  because Arm I8MM/KleidiAI quantize activations differently from x86 paths — §3.4).

## 3. Phase A — Platform abstraction and the llama.cpp/GGUF backend

### 3.1 Repository / branch setup
- Create `mac-m4` from `main`; commit the currently untracked working files first on
  `main` or the branch as appropriate (`energy_meter.py`, `README_arm.md`,
  `Doc/implementation_plan_benchmark.md`, `project_review.md`). Add `pm_fixture.plist`
  later (§8).
- Add `evol_inference/platform_info.py`: records, with every result file, `uname`, CPU
  model, **Arm feature flags** (`hw.optional.arm.FEAT_*` / `sysctl` on macOS; `/proc/cpuinfo`
  Features on Linux: `asimddp`, `i8mm`, `bf16`, `sve`, `sve2`, `sme`), core topology
  (P/E counts), memory, llama.cpp commit hash + CMake flags, KleidiAI version, power-mode
  (Low Power Mode / thermal state on macOS). Result JSONs without this block are
  rejected by the report script.

### 3.2 Backend interface (refactor, behavior-preserving)
Today `FitnessEvaluator` is welded to `WeightBank` + CUDA. Introduce thin interfaces:

- `Backend.prepare(genome) -> handle` — materialize a genome.
- `AccuracyProbe.perplexity(handle) -> float`
- `SpeedProbe.measure(handle) -> SpeedResult` (prefill tok/s, decode tok/s, repeats, spread)
- `EnergyProbe.measure(handle) -> EnergyResult` (J, avg W, net J after idle subtraction, J/token)

Implementations:
- `BnbGpuBackend` — wraps the existing `WeightBank`/`measure_latency_ms` (no behavior
  change; existing tests stay green and gate this refactor).
- `LlamaCppBackend` — new (below).
- `SimulatedArmProbe` — synthetic (§2.2.5).

`FitnessEvaluator` keeps its cache/normalization logic and gains the multi-objective
result type (§4). `FitnessResult` stays backward-compatible (defaulted new fields) so
old snapshots still load.

### 3.3 GGUF genome assembler (the replacement for module substitution)
Naively, each GA evaluation would run `llama-quantize` on a 7.6 GB F16 GGUF (tens of
seconds to minutes, plus a ~2–7 GB write per candidate). Instead, mirror `WeightBank`'s
idea — precompute once, assemble cheaply:

1. **Precompute**: for each precision in the alphabet, produce one uniform GGUF with
   `llama-quantize` (same imatrix, if any, for all). Quantization is per-tensor, so a
   tensor's bytes are identical whether it came from a uniform or a mixed file.
2. **Assemble**: a Python `GgufAssembler` (using `gguf-py`, shipped with llama.cpp) writes
   a genome GGUF by copying, for each of the 32 transformer blocks, the `attn_qkv`,
   `attn_output`, `ffn_up` (fused gate/up in Phi-3), `ffn_down` tensors from the source
   file selected by that block's super-block gene; embeddings, norms and output head come
   from a fixed choice (default: Q8_0 or F16 — a *documented constant*, not a gene, to
   match the current genome). Tensor-name ↔ `LINEAR_NAMES` mapping is a single table with a test.
3. Fallback/verification path: `llama-quantize --tensor-type 'blk\.(0|1|2|3)\..*=q8_0' …`
   (per-tensor type overrides) for a handful of genomes, to prove the assembler output
   is **bit-identical** (hash of tensor data) to llama.cpp's own mixed quantization.
4. Disk/RAM: keep the assembled file in tmpfs/RAM-disk where possible; one working
   file reused (overwritten) rather than one per genome; page-cache-friendly.

Risks to verify early (T0, an afternoon): exact tensor names for the Phi-3 arch in
llama.cpp; whether llama.cpp's Phi-3 support needs the `LongRoPE`/4k variant handled;
whether quantization imatrix is used; Q4_0 block alignment (tensor row length must be a
multiple of the block size — true for Phi-3's 3072/8192 dims).

### 3.4 KleidiAI and kernel-path attribution
- Build with `-DGGML_CPU_KLEIDIAI=ON` on Arm targets; `-DGGML_METAL=OFF` / `-ngl 0` on
  the Mac so the CPU (NEON/I8MM/SME via KleidiAI) does the work, not the GPU.
- **Important to verify, not assume**: which GGUF types KleidiAI actually accelerates
  (documented support has centered on Q4_0 and, in newer releases, Q8_0/F16 paths) and
  on which ISA levels (dotprod/I8MM/SME/SME2). Phase A includes a **kernel-path probe**:
  run each genome with `GGML_KLEIDIAI` on/off and record which backend op implementations
  were selected (llama.cpp's startup/`--verbose` backend logs, `llama-bench` JSON). The
  fitness result stores `kleidiai: on/off` — the A/B (KleidiAI vs. stock NEON `repack`)
  is itself a headline Arm result and an explanatory variable for why a genome is
  fast.
- Accuracy numerics differ by path (activation quantization to Q8 in I8MM/SME kernels).
  Therefore **final accuracy numbers are always re-measured on the target** (§4.3,
  §2.3), even though search-time accuracy may come from a cheaper machine in a dev-only
  mode.

### 3.5 Workload definition (decode matters on the edge)
The 3080 results used a single prefill-style 2048-token forward pass, and the README
flags decode as the unmeasured case. On Arm edge inference, decode (memory-bandwidth
bound, token-by-token) is the user-visible metric. Define two fixed probes using
`llama-bench` (or the llama.cpp library via `llama-cpp-python` if a C API is wanted):

- **Prefill**: `pp512` (and keep the 2048-token perplexity slice for accuracy).
- **Decode**: `tg128` with a fixed prompt, greedy sampling, fixed seed.
- Report both tokens/s numbers; fitness can use a configurable blend (default: decode
  tok/s, since that is the edge-relevant number; prefill reported and optionally weighted).
- Batch size 1 and a documented thread count (§5.3). Context length fixed (e.g. 512).

## 4. Phase B — Multi-objective fitness (accuracy / speed / power)

### 4.1 Objectives (all fractions of a fixed FP16-or-reference baseline measured on the same machine)

| Objective | Raw measurement | Normalized term |
|---|---|---|
| Accuracy | WikiText-2 perplexity, fixed 2048-token slice (as today) | `accuracy_penalty = (ppl − ppl₀)/ppl₀` (minimize) |
| Speed | decode tok/s (+ optional prefill tok/s) | `speed_gain = tps/tps₀ − 1` (maximize) — or `1 − latency/latency₀` to stay comparable with the README definition; pick one and keep the README's convention (§4.4) |
| Power | net energy per generated token (J/token) = (E_total − idle_W × t)/tokens | `energy_gain = 1 − (J/tok)/(J/tok)₀` (maximize) |

Derived reported-only metrics: tokens/s/W, energy-delay product (EDP = J/token × s/token),
avg W, peak W, model size (GB, still recorded — it's the proxy the README showed to be
misleading).

Baseline = the F16 genome on the same device, measured at the start of every run and
stored in the result file. If F16 does not fit comfortably (it does for 3.8B in 24 GiB)
this stays the baseline; for larger models use Q8_0 and say so.

### 4.2 Fitness forms (two, both implemented)
1. **Scalarized** (continuity with the README):
   `fitness = w_s·speed_gain + w_p·energy_gain − w_a·accuracy_penalty`, weights on the CLI.
   Reuses the existing steady-state GA unchanged.
2. **Pareto (preferred headline)**: NSGA-II-style non-dominated rank + crowding distance
   as the steady-state *eviction/selection key* (replace "evict worst scalar" with
   "evict worst rank, ties by crowding"), returning the 3-D Pareto front instead of a
   single point. `pareto.py` already exists for 2-D fronts; extend it to N-D and add
   hypervolume as the run-quality metric. The scalarized runs remain as cross-checks
   (a weight sweep should land on the Pareto front — a built-in consistency test).

   Note for the steady-state design: Pareto rank is *population-relative* — exactly
   the property the README says breaks cached fitness. Resolution: the **cache stores
   only raw objective vectors** (pure per-genome measurements); rank is recomputed from
   the vectors whenever the population changes (`refresh_from_cache` already exists for
   a similar purpose). Add a test asserting no rank is ever cached.

### 4.3 Handling noisy objectives (new, and a real engineering point)
Latency and energy are not pure functions of the genome; the README measured 3–7%
run-to-run latency spread, and power is noisier still on a laptop-class SoC (thermals,
DVFS, E-core scheduling).
- Each probe returns median + spread (MAD) over n repeats; cooldown/idle wait between
  candidates (monitor thermal state; abort/flag on throttling).
- Idle baseline measured once per batch (per `README_arm.md`) **and** re-measured
  periodically; drift beyond a threshold is logged.
- **Re-measurement of elites**: any genome that enters the Pareto front / top-k is
  re-measured ≥2 more times before it can be reported; final reported values are
  from a fresh interleaved A/B/A run of front members vs. baselines (reduces drift bias).
- Dominance comparison uses a noise margin (don't evict on differences below the
  measured spread).
- **Synthetic-data guard**: any result produced with `SimulatedArmProbe` or the `mock`
  meter is stamped `synthetic=true`; report scripts refuse to plot it as measured.

### 4.4 Experiments this enables (each needs a stated hypothesis before running)
1. Latency-optimal vs. energy-optimal genomes: do they differ on M4?
2. Does the bytes-proxy mislead on Arm the way it did on the 3080? (Hypothesis: decode
   is bandwidth-bound, so on M4 smaller *is* faster — the opposite of the 3080 finding.
   Reporting that reversal, with the roofline to explain it, is the cross-hardware lesson.)
3. Are the edge-super-blocks (0 and 7) still the sensitive ones under Q8_0/Q4_0?
4. KleidiAI on vs. off: effect on the Pareto front.
5. 2–3 seeds at the interesting weightings (the README's own review flagged the
   single-seed limitation).

## 5. Phase C — Search-space extensions (small, optional, high value for the Arm story)

### 5.1 Keep the baseline space first
Reproduce the 8×3 search before adding anything; it is the controlled comparison.

### 5.2 Wider precision alphabet (optional)
Add `Q4_K_M`/`Q5_K`/`Q6_K` or `IQ4_NL`. Cost: extra uniform source GGUF + support in the
assembler. Benefit: shows Arm-specific trade-offs (K-quant superblock dequant on NEON vs.
Q4_0's repack-friendly layout under KleidiAI). Gate: only if time remains after §5.3.

### 5.3 Runtime-configuration genes (recommended)
Co-search system parameters with the quantization genome:
- `n_threads` ∈ {1…10 on M4: P-cores only vs. P+E} — thread pinning is unavailable on
  macOS, so only the count is controlled (per `README_arm.md`); note macOS QoS may place
  threads on E-cores regardless and record that as a finding.
- `n_ubatch`/batch size for prefill.
This yields a genuinely systems-level result ("best quantization depends on thread
count because P/E cores differ in bandwidth and power") and fits the role's
"kernel level to system level" language. Implemented as extra genes with their own
mutation operator; cache key extended accordingly.

## 6. Phase D — Kernel-level work (the "Triton/CUDA/kernel programming" requirement)

The GA exercises kernels via llama.cpp, but the role asks for *writing* them. Two
bounded tracks; neither is on the critical path of the GA.

### Track A — read, attribute, explain (mandatory)
Per-genome kernel-path attribution (§3.4), plus a one-page analysis of how llama.cpp +
KleidiAI lays out Q4_0/Q8_0 weights (repacking, microkernel tile shapes, SDOT/I8MM vs.
SME2 paths) and how that predicts the measured speed differences.

### Track B — one original micro-kernel (strongly recommended)
- `kernels/arm/`: int8 (and int4-unpack) **GEMV** (decode-shape: M=1) in C++ with NEON
  intrinsics using `SDOT` and, where available, `I8MM` (`SMMLA`); scalar reference and
  tests; CMake with feature-gated builds.
- **Correctness on x86 desktop**: cross-compile (`aarch64-linux-gnu-g++` or clang
  `--target=aarch64`) and run under **QEMU user-mode** (`-cpu max`) — bit-exact against
  the scalar reference and a NumPy oracle. (Install: `qemu-user`, `g++-aarch64-linux-gnu`;
  not currently on this machine.)
- **Performance**: Google Benchmark / custom harness; run on Graviton (T3) and M4 (T4);
  compare to KleidiAI's microkernel and to llama.cpp's own at the same shape; report
  achieved GB/s vs. the machine's STREAM bandwidth (roofline position).
- **Same-op GPU counterpart** on the 3080: a Triton (and/or CUDA) int8/int4 GEMV, to
  evidence the Triton/CUDA requirement and to give a cross-architecture comparison
  (tile choice, memory coalescing vs. cache-line/prefetch behavior). Developed and tested
  entirely on the desktop.
- Stretch: SVE/SVE2 variant tested on Graviton, SME2 variant on M4 (apple clang
  exposes SME intrinsics; macOS SME needs streaming-mode care — verify toolchain
  support before committing).

## 7. Phase E — Profiling and analysis workflow (the "profiling tools" requirement)

- **M4 (T4)**: `powermetrics` (CPU/GPU/ANE power, per-cluster frequency, residency),
  Instruments (Time Profiler, CPU Counters where permitted), `sysctl` feature queries.
  Capture a representative profile for: F16, uniform Q8_0, uniform Q4_0, and the GA's
  best genome.
- **Graviton (T3)**: `perf stat`/`perf record` with PMU events (IPC, cache misses,
  memory-bandwidth-related events), flame graphs, and **Arm Performance Studio /
  Streamline** if the instance supports it (verify host/target requirements); Linux +
  same distro as the desktop makes this the easiest place to learn the tooling.
- **Desktop (T0/T1)**: `perf` (present), `nvidia-smi`/Nsight for the GPU kernel track,
  RAPL for x86 energy.
- Deliverable: a **roofline** per platform (STREAM bandwidth + peak int8 throughput from
  the micro-kernel) with prefill and decode points of each key genome plotted — the
  single most effective "I understand memory hierarchies" artifact.
- Deliverable: a **perf-issue write-up in the style of the role** ("diagnose and resolve
  performance challenges"): pick one anomaly the data shows (e.g. a precision that is
  smaller yet slower; thread scaling cliff when E-cores join) and walk through
  hypothesis → counters → cause → fix/recommendation.

## 8. Phase F — Energy harness completion (`energy_meter.py`)

Already written; remaining work, in order:
1. **Tests** (T0): unit tests for each backend — plist parser (replay), baseline
   subtraction maths, `Measurement` edge cases, mock determinism, RAPL wraparound
   (`max_energy_range_uj`), backend auto-selection. Put under `tests/test_energy_meter.py`.
2. **RAPL realism check** (T0): run `measure_cmd` on real llama.cpp x86 inference;
   verify idle-baseline subtraction yields stable net-J across repeats (CV target
   < ~5%); this de-risks the statistics before the Mac exists.
3. **Parser fixture**: first action on the Mac is `record pm_fixture.plist --seconds 10`;
   commit it; from then on `replay` tests the real format on Linux. Open risk (unverified
   key names `cpu_power`/`combined_power`/`cpu_energy`) is closed there. Consider
   also capturing GPU/ANE power keys to *prove* they stay idle during CPU-only runs
   (a sanity check that Metal is truly off).
4. **Sampling vs. run length**: powermetrics at 100 ms with short runs gives few samples;
   set minimum measurement duration (e.g. ≥ 20 s per candidate, looping the benchmark)
   and report sample count; test on RAPL/mock first.
5. **Integration**: `EnergyProbe` wraps `measure_cmd(llama-bench …)` so each candidate's
   speed and energy come from the *same* run (avoids doubling Mac time and measuring
   different thermal states).
6. Optional free CI: GitHub Actions macOS arm64 runner builds llama.cpp and runs T0/T2-like
   tests; try `record` once (likely blocked in a VM).

## 9. Phase G — Mac session readiness (everything done before the instance starts)

Deliverables, all tested at T0–T3 first:
- `scripts/bootstrap_mac.sh` (thin; installs Homebrew, cmake, python, builds llama.cpp
  with the flags above, creates venv) — keep logic in Python `scripts/bootstrap_check.py`
  so it can also run on Graviton/Linux for the same checks.
- **Dry-run on Graviton** (T3): the same bootstrap + a short GA run with the real
  `LlamaCppBackend` and `rapl`-less/`mock` energy, proving the Arm Linux build,
  KleidiAI, assembler, and result schema work on genuine Arm silicon for cents.
- Pre-staged artifacts: the uniform per-precision GGUFs (F16/Q8_0/Q4_0) and tokenizer
  in S3 in the same region (≈ 7.6 + 4.1 + 2.2 GB for Phi-3-mini); checksums verified.
- `scripts/run_queue.py`: unattended queue (seeds × weightings × configs), resumable,
  snapshot every N evaluations (reuse `save_snapshot`), per-run logs, **budget guard**
  (stop launching new runs when remaining wall-clock < one run), results synced to S3
  periodically so a crashed/ended instance loses nothing.
- **Selftest gate**: `python energy_meter.py selftest` + a 3-genome smoke evaluation +
  platform record must pass before the queue starts; failure aborts early (don't burn the day).
- Run order for the 24 h block: (1) selftest + fixture recording + platform info;
  (2) baselines (F16/Q8_0/Q4_0/heuristic) with full probes ×N repeats; (3) thread-count
  scan on uniform Q8_0/Q4_0 (cheap, informs gene range); (4) main GA seeds; (5) KleidiAI
  off ablation; (6) profiling captures; (7) final interleaved re-measure of the Pareto front.
- Rough budget (to be replaced by Graviton-measured numbers): per evaluation ≈ assemble
  (seconds) + perplexity (tens of seconds on CPU) + speed/energy probe (≥ 20–30 s) ≈
  ~1 min → ~600–1,000 evaluations per search → plan ≈ 2–3 searches per day. If
  accuracy evaluation dominates, the perplexity slice can be shortened *only* with a
  recorded justification, since it changes the fitness definition.
- Provider choice (AWS vs. Scaleway M4, ~1/8 price per `README_arm.md`): nothing in the
  code should depend on the provider; note SIP/passwordless-sudo setup for `powermetrics`
  in the bootstrap.

## 10. Phase H — Reporting and portfolio deliverables

- `README_arm.md` becomes the Arm-port README (rewritten from handoff notes): problem,
  method, hardware table (Mac, Graviton, desktop), results, honest limitations.
- Plots: 3-D/pairwise 2-D Pareto fronts (accuracy–speed, accuracy–energy, speed–energy),
  hypervolume vs. evaluations, roofline, thread-scaling, KleidiAI on/off.
- A "porting a GPU quantization experiment to Arm: lessons" write-up (what didn't port,
  what did, which assumptions the data overturned) — the customer-engineering voice
  the role asks for.
- Update the presentation (`presentation/`) with an Arm section; keep the same
  null-results-reported-honestly convention.
- Optional: an ExecuTorch export of the best genome with the XNNPACK/KleidiAI delegate
  as a second Arm-ecosystem runtime comparison (reuses the compiled-runtime benchmark
  plan's thinking; scope only if schedule allows — mixed per-block precision support is the
  open question).

## 11. Testing plan (desktop-first)

New tests (all runnable at T0 unless noted):
- `test_gguf_assembler.py`: tensor-name table; assembled-file tensor hashes equal the
  source tensors; assembled file loads in llama.cpp (T1) and gives expected perplexity;
  bit-identical to `llama-quantize --tensor-type` output on sample genomes (T1).
- `test_backends.py`: `BnbGpuBackend` parity with current behavior (T1); backend
  interface contract tests with the simulated backend (T0).
- `test_fitness_multiobjective.py`: normalization, cache stores raw vectors only,
  noisy-measurement re-measure policy, synthetic-data guard.
- `test_pareto_nd.py`: N-D dominance, rank, crowding distance, hypervolume (vs. known
  analytic cases).
- `test_ga_pareto.py`: steady-state eviction by rank/crowding; scalarized sweep lands on
  the Pareto front.
- `test_energy_meter.py` (§8.1).
- `test_platform_info.py`: parsing of `sysctl`/`/proc/cpuinfo` fixtures (include a captured
  Mac sample once available).
- T2 smoke tests (marked `@pytest.mark.arm_emulated`, off by default): aarch64 build,
  proxy-model perplexity within tolerance of x86, micro-kernel bit-exactness under QEMU.
- Existing 63 tests must stay green after the refactor — run them at the end of every phase.

Pytest markers: `gpu`, `arm_emulated`, `arm_native`, `mac_only`, so each tier runs only
its own set.

## 12. Order of work and rough effort

| Step | Phase | Tier | Est. (focused days) |
|---|---|---|---|
| 1 | Branch, platform_info, backend interfaces, refactor (tests green) | T0 | 1–2 |
| 2 | llama.cpp builds (x86 CPU + CUDA); tensor-name check; GGUF assembler + equivalence tests | T0/T1 | 2–3 |
| 3 | Sensitivity table + cross-backend consistency (§2.3) on the 3080 | T1 | 1 |
| 4 | Multi-objective fitness, N-D Pareto, GA eviction change, simulated probe | T0/T1 | 2–3 |
| 5 | Energy harness tests + RAPL realism; speed+energy probe integration | T0/T1 | 1–2 |
| 6 | Full dry-run GA on the 3080 (real accuracy, simulated Arm cost) | T1 | 1 |
| 7 | Arm emulation (QEMU) build + proxy-model correctness | T2 | 1–2 |
| 8 | Micro-kernel (NEON/SDOT/I8MM) + Triton/CUDA counterpart | T0/T1/T2 | 3–5 |
| 9 | Graviton dry-run: bootstrap, KleidiAI, perf/Streamline workflow, timing numbers for budget | T3 | 1–2 |
| 10 | Mac readiness (bootstrap, staging, queue, selftest gate) | T0–T3 | 1–2 |
| 11 | **Mac 24 h block(s)** | T4 | 1–2 days of instance time |
| 12 | Analysis, roofline, write-ups, presentation | — | 3–4 |

Total ≈ 3–5 weeks part-time; steps 8 and the ExecuTorch option are the first things to
cut if the schedule tightens (the GA port, multi-objective fitness and measured M4
results are the core).

## 13. Risks and open questions

| Risk | Mitigation / how to resolve cheaply |
|---|---|
| llama.cpp tensor names / Phi-3 arch quirks break the assembler | Inspect GGUF with `gguf-py` at T0 before designing further; fall back to `llama-quantize --tensor-type` per genome if splicing proves brittle (slower, still feasible) |
| KleidiAI doesn't accelerate all alphabet types (maybe only Q4_0) | Kernel-path probe (§3.4); turn it into a finding, and keep stock-NEON as the comparison |
| Energy readings are noisy / powermetrics keys differ | RAPL realism tests; fixture on first Mac minute; repeats + noise margin (§4.3) |
| Thermal throttling / DVFS contaminates ordering | Cooldown, interleaved re-measure, thermal state logging; report spread |
| macOS can't pin threads; E-cores intrude | Treat thread count as the only knob; study P-only vs. P+E explicitly (§5.3) |
| Accuracy differs between x86 and Arm kernels | Always re-measure final accuracy on target; quantify gap at T2/T3 |
| Mac budget burn from avoidable bugs | Tier promotion rule; Graviton dry-run; selftest gate; budget guard |
| Scope creep (kernels, ExecuTorch, wider alphabet) | Core = Phases A, B, F, G; C/D/E extras are explicitly cuttable |
| Spurious claims from synthetic probes | `synthetic=true` stamping and report-script refusal |
| Facts in this plan about KleidiAI/SME/powermetrics keys/instance availability are from memory | Each is tagged "verify"; first T0/T3/T4 task is to confirm before depending on it |

## 14. Decisions (resolved)

The five open questions from the first draft were resolved by taking the recommendations,
with the model choice made explicit. Rationale kept here so the plan is self-contained.

### D1. Speed objective = decode tokens/s (`tg128`); prefill measured and reported
Prefill is compute-bound (kernel and dequant overhead dominate; this is the regime of the
3080 "INT4 is slowest" result). Decode is memory-bandwidth-bound and is what a user feels
on an edge device. Search optimizes decode; prefill (`pp512`) comes from the same
`llama-bench` run and is reported as an ablation. It tests the hypothesis that on Arm
decode is bandwidth-bound, so smaller *is* faster — the opposite of the 3080 finding.
If that hypothesis fails, escalate to prefill and decode as separate objectives (4-D).

### D2. Pareto is the headline; scalarized is the cross-check; build scalarized first
Scalarized weight sweeps cost one run per weight triple (3 objectives => many runs on a
24 h Mac block) and miss non-convex front points. One Pareto run returns the whole front.
Build order: scalarized path first (nearly free, exercised on the 3080), then N-D Pareto
ranking, which becomes the headline. Scalarized runs must land on the front (consistency
test). The cache stores raw objective vectors only; rank is recomputed (§4.2).

### D3. Runtime genes are out of v1; do a separate thread-count scan instead
Uniform Q8_0 and Q4_0 at 1..10 threads (about an hour of Mac time) fixes the thread count
for the search and gives the P/E-core finding without enlarging the search space ~5x or
confounding genes with macOS scheduling noise. Runtime genes (§5.3) become a follow-up.

### D4. Kernel-track scope: Track A + NEON/SDOT/I8MM micro-kernel + Triton/CUDA counterpart
Deferred: SVE2/SME2 variants and ExecuTorch (toolchain risk / possibly a different experiment).

### D5. Multiple models, to test whether results generalize

| Tier | Model | Params / layers | Role |
|---|---|---|---|
| 1 | Phi-3-mini-4k-instruct | 3.8B / 32 | Continuity with all existing results; fused qkv and gate_up |
| 1 | Llama-3.2-3B(-Instruct) | 3.2B / 28 | Different family: separate q/k/v + gate/up, GQA. Similar size, so similar Mac cost |
| 2 | **Llama-3.1-8B(-Instruct)** | 8B / 32 | **Scale test.** 32 layers match the 8x4 grouping exactly. Baseline is Q8_0 (F16 is 16 GB: tight on 24 GiB and impossible on the 10 GB 3080). ~2x Mac time per evaluation, so baselines, sensitivity table and a shorter search |
| all | every model | — | Cheap 3080 sensitivity table (~20 evaluations per model) |

Design consequences:
- A `ModelSpec` per model (`evol_inference/model_spec.py`): layer count, super-block
  grouping, GGUF tensor-name map, reference (baseline) precision, HF repo id. The genome is
  **always 8 genes**, with near-equal groups for models whose layer count is not divisible
  by 8 (28 layers -> 4,4,4,4,3,3,3,3). Per-block parameter counts are tracked, so uneven
  groups are handled exactly.
- Models are compared by **relative depth** (block 0 vs. block 7), never raw layer index.
- Llama models are gated on Hugging Face (licence acceptance needed before download).
- Assembler and its tests are per-architecture. Pre-stage per-model GGUF sets in S3
  (Llama-3.1-8B: F16 ~16 GB, Q8_0 ~8.5 GB, Q4_0 ~4.5 GB).

**Pre-registered definition of "results are similar"** (fixed before any Arm run, so the
comparison cannot be argued either way afterwards):
1. Per-block sensitivity profiles (by relative depth) have high rank correlation
   (Spearman), and the edge blocks are the most sensitive in each model.
2. The baselines rank the same way on all three objectives.
3. The Pareto fronts overlap (hypervolume ratio and front-membership of the shared
   uniform/edge-protected patterns).
4. The weighting at which a mixed genome first beats the best uniform one falls in the same range.
5. The discovered genomes have the same shape (which blocks stay at higher precision).
Disagreement on any of these is a reported result, not a failure. With 3 models,
"similar" is evidence, not proof; the write-up says so.

### Remaining question for the author
None blocking T0. (Open for later: whether to also run Qwen2.5-3B as a fourth architecture.)

## 15. T0 implementation status (branch `mac-m4`)

T0 = tier T0 of §2.1: everything that runs on any host with no GPU, no llama.cpp binary
and no Arm hardware. **Done** (all with tests; run `PYTHONPATH=. .venv/bin/python -m pytest`):

| Item | Module | Tests |
|---|---|---|
| Genome split from bitsandbytes (`Precision`, `Genome`, `N_SUPER_BLOCKS` now in `genome.py`; `weight_bank` re-exports, so old imports and snapshots are unchanged; `fitness.py` no longer imports bnb at runtime) | `evol_inference/genome.py` | existing 63 |
| Test suite no longer needs CUDA: only tests using `model`/`bank`/`tokenizer` are auto-marked `gpu` and skipped without CUDA (was: whole suite skipped); `pytest.ini` with `pythonpath` and tier markers | `tests/conftest.py`, `pytest.ini` | — |
| `ModelSpec` for phi3-mini, llama3.2-3b, llama3.1-8b; uneven grouping (28 -> 4,4,4,4,3,3,3,3); GGUF tensor classification | `model_spec.py` | `test_model_spec.py` |
| GGUF genome assembler (copy pre-quantized tensor bytes per gene; metadata preserved; exact `bytes_for`, per-block param counts; validation of tensor sets/shapes/spec) | `gguf_assembler.py` | `test_gguf_assembler.py` (synthetic Q8_0/Q4_0/F16 GGUFs) |
| N-D Pareto: dominance with noise margin, non-dominated sort, crowding, exact hypervolume | `pareto_nd.py` | `test_pareto_nd.py` (incl. Monte-Carlo HV check) |
| Objective vector, fixed-baseline normalization, scalar weights | `objectives.py` | `test_mo_search.py` |
| Synthetic accuracy/Arm probes (`synthetic=True`), probe protocols | `probes.py` | `test_mo_search.py` |
| Multi-objective evaluator: raw-vector cache only, `remeasure` for elites, synthetic guard | `mo_fitness.py` | `test_mo_search.py` |
| Steady-state GA with `ScalarRanker` / `ParetoRanker`, hypervolume stagnation stop, snapshots (synthetic snapshots refused by measured runs) | `mo_ga.py` | `test_mo_search.py` |
| Platform record + `require_platform` (Arm feature parsing for macOS `sysctl` and Linux `/proc/cpuinfo`, M4 and Graviton3 fixtures) | `platform_info.py` | `test_platform_info.py` |
| llama.cpp probes with injectable runner: `llama-perplexity` and `llama-bench` (prefill/decode in separate energy windows; `total` vs `differential` energy method; mock/replay meters stamp results synthetic) | `llama_probes.py` | `test_llama_probes.py` (captured-output fixtures) |
| `energy_meter.py` test suite: parser, baseline maths, RAPL wraparound, mock, replay, powermetrics streaming against a fake process, backend selection | `energy_meter.py` | `test_energy_meter.py` |
| Synthetic end-to-end driver (`scripts/mo_search_synthetic.py`) | `scripts/` | manual run |

**Known gaps / things T0 cannot settle (carried to later tiers):**
- RAPL is root-only readable on this desktop (`energy_uj` is `-r-------- root`), so the
  real-RAPL test and the §8.2 realism check are skipped until read access is granted
  (e.g. `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`, which resets on reboot).
  Not changed automatically.
- GGUF tensor names in `model_spec.py` and the `llama-bench` / `llama-perplexity` output
  formats are from memory; fixtures encode that assumption. T1 must verify against a real
  llama.cpp build and real converted GGUFs, then correct the table/fixtures.
- Existing GPU path untouched functionally; the 63 original tests pass (run on the 3080
  before and after the refactor -- see the commit message / final status).
- Not yet done from the plan: T1 (llama.cpp CUDA/CPU builds, real GGUFs for all three
  models, sensitivity tables, cross-backend checks), real `EnergyProbe` integration runs,
  T2 QEMU, kernels, Mac readiness.

## 16. T1 status (steps 1-3 done)

Environment: llama.cpp `ec7630a` (shallow clone at `~/work/llama.cpp`, outside this repo),
two builds: `build-cpu` and `build-cuda` (`-DCMAKE_CUDA_ARCHITECTURES=86`; system nvcc 12.0
rejects gcc 13, so the CUDA build uses `g++-12` as host compiler). Converted GGUFs live in
the git-ignored `models/gguf/`: `phi3-mini-f16.gguf`, and uniform `...-q8_0-pure.gguf` /
`...-q4_0-pure.gguf` made with `llama-quantize --pure` (essential: without `--pure`,
llama.cpp's k-quant heuristics may give some tensors a different type than requested).

| Step | Result |
|---|---|
| 1. Builds + output formats | Both builds work; CUDA binaries run on the 3080. `llama-bench -o json` keys (`n_prompt`, `n_gen`, `avg_ts`, `stddev_ts`, `samples_ts`) and `Final estimate: PPL = ...` match the parsers. Log lines carry timestamps; the unanchored regex still matches. Real outputs saved as `*_real_ec7630a.*` fixtures. `llama-bench` JSON also reports `repack`, `backends`, `cpu_info`; `llama-perplexity`'s `system_info` line lists CPU features (`REPACK = 1`, NEON/I8MM on Arm), usable for kernel-path attribution (§3.4). |
| 2. Phi-3 GGUF + tensor names | Converted (195 tensors, 7.6 GB F16). `model_spec.py`'s table is **correct as written**: `attn_qkv`, `attn_output`, `ffn_up` (fused gate/up), `ffn_down`; norms are `*_norm`, F32. Phi-3 4k has no rope-factor tensors. |
| 3. Assembler vs `llama-quantize` | **Bit-identical, tensor for tensor,** for two mixed genomes (FP16/INT8/INT4 mixtures) against `llama-quantize --pure --tensor-type` overrides, including tensor types; a negative control (genomes differing in one gene) differs in exactly that gene's 16 tensors. The splice-assemble design is confirmed. `tests/test_llamacpp_t1.py` (marker `llamacpp`; skips if llama.cpp/GGUF absent; first run ~3 min because it quantizes the sources, cached afterwards). |

Early observations (a preview of step 4, not yet a result):
- llama.cpp perplexity on the 3080, one 2048-token chunk: F16 **4.5221**, Q8_0 **4.5324**
  (+0.23%). HF FP16 gave 4.3708 on the "same" slice, so absolute values are **not
  comparable across backends** (tokenization/BOS handling differs; plan §2.3). Fitness only
  uses penalties relative to the same backend's baseline, so this is fine, but the
  README must never mix the two.
- Q8_0's accuracy cost (+0.23%) is far smaller than bitsandbytes INT8's (+1.5%): the
  Q8_0 genes will be much cheaper than INT8 was on the GPU, which may move the weighting
  at which edge blocks stop being protected. Expect the GGUF sensitivity table to differ
  from the bnb one in magnitude, possibly in pattern.
- Exact bytes (genome tensors only): F16 6.75 GiB, Q8_0 3.59 GiB, Q4_0 1.90 GiB (block-scale
  overhead makes Q8_0 8.5 bits/weight and Q4_0 4.5, not 8 and 4).

**Next (T1 steps 4-7):** fixed-slice alignment check (same token IDs for HF and llama.cpp, or an
explicit decision to define the GGUF slice independently), `LlamaPerplexityProbe` as the real
accuracy probe, per-block sensitivity table for Phi-3, then the other two models (HF licence
acceptance needed for the Llama models) and the real-accuracy / simulated-Arm dry-run search.
