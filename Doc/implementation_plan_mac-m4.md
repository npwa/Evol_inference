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

### Step 4: evaluation slice and the real accuracy probe (done)

**Why llama.cpp gave 4.5221 where HF gave 4.3708 -- fully explained.**
- Tokenization is *identical*: Phi-3's HF tokenizer and llama.cpp produce the same IDs over
  the whole 341,468-token WikiText-2 test text.
- `llama-perplexity` scores only the **second half of each context chunk** (`first = n_ctx/2`
  in `perplexity.cpp`): for `-c 2048 --chunks 1` that is 1023 tokens, each with >=1024 tokens
  of context. The HF path scores all 2047. On the *same* 1023 tokens HF-FP16 gives **4.5229**
  vs llama.cpp **4.5217-4.5238**: 0.05% apart (the CUDA/CPU/HF fp16 numerics).
- `--ppl-stride` is not a way out: it silently switches to a different procedure (context
  3072) and gave nonsense (PPL ~37), so it is not used.

**Decision: use stock `llama-perplexity -c 2048 --chunks 1`; the GGUF slice is "the second half
of the first 2048-token chunk" (1023 scored tokens).** Same compute as the base project
(one 2048-token forward), no custom evaluator to maintain on the Mac. `--chunks N` is
available if more scored tokens are wanted (N chunks -> 1023N tokens, N forwards). Absolute
perplexities are comparable only within one window; the README's (full-window) numbers are
never mixed with these. Documented in `evol_inference/eval_data.py`.

Cross-backend table (`scripts/t1_crosscheck_ppl.py`, Phi-3-mini, uniform genomes; penalties
relative to each path's own FP16):

| path / window | FP16 | INT8 / Q8_0 | INT4 / Q4_0 |
|---|---|---|---|
| HF + bitsandbytes, all 2047 tokens (README) | 4.3708 | 4.4381 (+1.54%) | 4.6346 (+6.04%) |
| HF + bitsandbytes, second half (1023) | 4.5229 | 4.5646 (+0.92%) | 4.7097 (+4.13%) |
| llama.cpp CUDA, second half | 4.5217 | 4.5324 (+0.24%) | 4.8053 (+6.27%) |
| llama.cpp CPU (AVX512), second half | 4.5238 | 4.5350 (+0.25%) | 4.8046 (+6.21%) |

Findings:
1. **Same ordering everywhere** (FP16 < INT8 < INT4) -- the cross-backend consistency check of §2.3 passes.
2. **Window matters for penalties too, not just absolute values**: bnb INT8 costs +1.54% on all
   2047 tokens but +0.92% on the second half. The early tokens (little context) are the more
   quantization-sensitive ones, and the llama.cpp window never scores them. So per-block
   sensitivities measured with llama.cpp will be systematically smaller than the README's;
   compare backends only on the matched window (the HF path can score the same window).
3. **Q8_0 is far cheaper in accuracy than bnb INT8, while Q4_0 is worse than bnb NF4**
   (perplexity deltas above: +0.24% vs +0.92%; +6.27% vs +4.13%). *Caveat found in step 5:*
   perplexity deltas on a ~1000-token sample carry ~+-0.2% sampling noise, so the INT8 figures
   here are not reliable; the KL-divergence numbers in step 5 supersede them (Q8_0 +0.067% vs
   bnb INT8 +1.08%, ~16x; Q4_0 +6.85% vs NF4 +5.81%).
4. **CPU vs CUDA numerics differ by <=0.06%** in perplexity (x86 AVX512 vs 3080). Arm kernels
   (I8MM/SME activation quantization) will differ again; accuracy is therefore re-measured on the
   target for final results (plan §3.4), and this size of difference is the floor for the
   noise margin on accuracy comparisons across machines.
5. **Cost per accuracy evaluation on the 3080:** ~5 s assemble (warm page cache, 4.6 GiB file)
   + ~3 s `llama-perplexity` = **~8 s**. Cold-cache first touches of a source file inflate the
   very first evaluations to 16-44 s. A 650-evaluation search is therefore ~1.5 h of accuracy
   probing on this machine (the base project's whole search took 10 min), so assembly cost
   is now the dominant term; a possible optimization is an in-place incremental assembler
   (rewrite only the tensors whose gene changed), deferred until the measured Arm per-eval
   cost shows whether it matters.

Code: `AssembledGgufProvider` (one reused work file, skips identical genomes, invalidated on
failed assembly), `eval_data.py` (WikiText-2 text identical to the HF path's concatenation;
`scored_tokens`), `LlamaPerplexityProbe(chunks=...)`; real-probe tests in
`tests/test_llamacpp_t1.py` (deterministic, ordering FP16<INT8<INT4, mixed genome strictly
between its uniform bounds, FP16 within 0.5% of the HF second-half value).

## 17. T1 step 5: per-block sensitivity (done) -- and a change of accuracy metric (D6)

### 17.1 Perplexity deltas cannot resolve per-block sensitivity
The first sweep (16 single-block genomes + uniform, perplexity delta on the 1023-token window)
was noise: on the *bitsandbytes* path several single-block penalties came out **negative**
(-0.17%, -0.06%: quantizing a block "improved" perplexity), the sum of single-block penalties
disagreed with the uniform genome (INT8: 0.82% vs 0.24% on llama.cpp; 1.34% vs 0.92% on bnb)
and the two backends' profiles were uncorrelated (Spearman -0.33 / -0.24). A perplexity *difference*
on ~1000 tokens measures how a specific random quantization-error realization happens to
interact with those tokens (llama.cpp reports PPL ratio 1.0038 +- 0.0023, i.e. ~60% relative
error on a 0.4% effect), not the mean damage.

**Consequence beyond this table:** any conclusion drawn from small perplexity differences on a
single fixed slice -- including the README's per-block sensitivity numbers and GA outcomes
at INT8 -- has this noise inside it, and a GA that optimizes the slice can fit it.

### 17.2 D6: accuracy objective = KL divergence to the original model's logits
`llama-perplexity --kl-divergence` compares each candidate's token distributions to saved logits
of the original model. Measured on the same 1023 tokens, mean KLD has ~1-5% relative standard
error (e.g. 0.00165 +- 0.00009) and a numerical floor of 1e-5 nats. It is deterministic,
non-negative, and additive-friendly.
* **Definition:** `accuracy_penalty = exp(KLD - KLD_ref) - 1` -- the expected fractional perplexity
  increase, in the base project's units (so existing weights keep their meaning).
  Implemented as `LlamaKldProbe.perplexity = ppl0 * exp(KLD)` feeding the unchanged evaluator.
* **Base = the original model** (F16 GGUF; Q8_0 for Llama-3.1-8B), not an assembled genome.
  The assembled all-FP16-genome is *not* the original: its embeddings/output head are Q8_0
  (`fixed_precision`). Measured: that costs **0.000985 nats**, about half of uniform Q8_0's total
  KLD on Phi-3, i.e. the fixed tensors are a significant constant that the reference must not hide.
* Raw perplexity is still logged for continuity; fitness uses KLD.
* Cost: same single 2048-token forward per genome + saving the base logits once (65 MB per
  chunk). `--chunks N` is available if tighter error bars are wanted.

### 17.3 Phi-3-mini sensitivity table (llama.cpp CUDA, KLD, 1023 scored tokens)
Penalty % = expm1(KLD) with only that block quantized (rest at the FP16-genome reference):

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | sum | uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 0.011 | 0.009 | 0.009 | 0.006 | 0.005 | 0.007 | 0.006 | 0.016 | 0.068 | 0.067 |
| Q4_0 | 0.566 | 0.656 | 0.739 | 0.960 | 0.654 | 0.502 | 0.420 | 1.431 | 5.93 | 6.85 |
| bnb INT8 | 0.413 | 0.202 | 0.299 | 0.142 | 0.087 | 0.045 | 0.027 | 0.042 | 1.26 | 1.08 |
| bnb NF4 | 0.544 | 0.612 | 0.763 | 0.808 | 0.525 | 0.523 | 0.422 | 1.284 | 5.48 | 5.81 |

Findings (each backed by the table; `results/sensitivity_phi3-mini*_kld.json`):
1. **Q8_0 is almost exactly additive** (sum 0.068% vs uniform 0.067%): per-block costs are
   independent, so the separability model of the README holds for it. **Q4_0 is mildly
   super-additive** (+15%: errors compound across blocks), as is NF4 (+6%).
2. **INT4 profiles replicate across backends: Spearman 0.976** (Q4_0 vs NF4). Both put the
   largest cost on the *last* block (24% of the total), block 3 next, and the least on
   blocks 5-6. This is an independent confirmation of a real, quantizer-agnostic structure.
3. **INT8 profiles do not agree (Spearman 0.35)** because the quantizers differ: bnb LLM.int8 is
   front-loaded (block 0 = 33%, blocks 0-3 = 84%, last block 3%), whereas Q8_0 costs are tiny
   and flat with a slight last-block bump (23%). Q8_0 has a total cost 16x lower than bnb INT8.
4. **Revision of README finding 3** ("block 0 alone accounts for ~70% of uniform INT8's cost,
   ~30x an interior block"): on the KLD metric block 0 is indeed the most sensitive for bnb
   INT8, but it carries **~33%, about 3-5x an interior block**, not 70% / 30x; the original
   estimate came from perplexity deltas now known to be noisy at this scale. The direction
   (front-loaded INT8 sensitivity on bnb; edges matter) survives; the magnitude does not.
   README to be amended when the Arm results are written up (not edited yet).
5. **Hypothesis for the search on Arm** (to test, not a result): with Q8_0 near-free and Q4_0
   expensive, the interesting Pareto structure is Q4_0 vs Q8_0 per block, with the last block
   (and block 3) the first to be held at Q8_0 as the accuracy weight rises.

### 17.4 Reproducibility, hardware transfer, speed
* The CUDA table was reproduced bit-for-bit across two independent sweeps (deterministic).
* **Hardware transfer (CUDA vs CPU/AVX512 build, same GGUFs, same metric):** the **Q4_0 profile is
  identical** (Spearman 1.00, largest per-block difference 0.016 percentage points; uniform
  6.85% vs 6.88%). The Q8_0 profile agrees closely in magnitude (differences <= 0.0023 points,
  uniform 0.067% vs 0.060%) but its *ranking* is only partly stable (Spearman 0.81): the values
  sit at ~1e-4 nats, within an order of magnitude of the 1e-5 numerical floor. So per-block
  Q8_0 ordering must not be over-read; Q4_0 structure is hardware-independent.
* **One real hardware dependence:** the cost of the *fixed* Q8_0 tensors (embeddings + output head)
  is 0.000985 nats on CUDA but 0.000175 on CPU. The same Q8_0 bytes give different accuracy on
  different kernels (CUDA's MMQ path quantizes activations to Q8_1; the CPU path differs). This
  is exactly why final accuracy must be measured on the Arm target (plan §3.4): Arm's I8MM/SME
  paths will add their own, and the constant fixed-tensor term should be expected to move.
* Cost: the CPU sweep took 900 s (~47 s per genome on 8 threads, dominated by the 2048-token
  CPU forward), vs ~130 s on the 3080.
* Work-file placement matters: assembling onto the NVMe disk ran at ~50 MB/s under memory
  pressure (each ~4-7 GB assembly took ~45 s, dirty-page write-back throttling), vs ~5 s when
  the work file lives in RAM (`/dev/shm`). The 17-genome sweep takes ~2 minutes in RAM.
  `scripts/t1_sensitivity.py` defaults to `/dev/shm`. On the Mac, use a RAM disk or confirm
  APFS write speed during the bootstrap selftest.

Code: `sensitivity.py` (table, additivity residual, ranking, Spearman, edge dominance),
`LlamaKldProbe` + `parse_kld`, `SimulatedAccuracyProbe.from_table` (synthetic runs can now use
measured sensitivities), `scripts/t1_sensitivity.py` (`--metric ppl|kld`, `--hf-bnb`, `--build`).

## 18. T1 step 6: the other models (Llama-3.1-8B done; Llama-3.2-3B blocked on licence access)

### 18.1 Access and downloads
* The supplied read-only Hugging Face token authenticates, and **Llama-3.1-8B-Instruct is readable**
  (15 GB safetensors, downloaded to `models/llama-3.1-8b-instruct`, git-ignored).
* **Llama-3.2-3B-Instruct returns 403 (`GatedRepoError`, gating = manual)**: the account has not been
  granted access to that repo. It needs the licence accepted on its model page (and Meta's manual
  approval). Nothing was downloaded for it; it is the only part of step 6 still open.
* The token was never written to a file in the repo (passed as an environment variable per command).

### 18.2 Llama-3.1-8B pipeline
* Converted to F16 GGUF (16 GB, 292 tensors) and made uniform `--pure` Q8_0 (8.5 GB) and Q4_0 (4.5 GB)
  from the F16 (re-quantizing Q4_0 *from Q8_0* would double the error, so F16 is kept as the source).
* **Tensor names match `model_spec.py` exactly** (`attn_q/k/v/output`, `ffn_gate/up/down`; plus
  `rope_freqs.weight`, `token_embd`, `output`, `output_norm` outside the genome).
* **Assembler is bit-identical to `llama-quantize --tensor-type` on a different architecture**
  (separate q/k/v, GQA, 128k vocab): both mixed genomes, all tensors and types
  (`tests/test_llamacpp_t1.py` now parametrized over models; 8B cases are marked `slow`, ~5 min).
* **Fits on the 3080:** Q8_0 (8.5 GB) with `-ngl 99` runs a 2048-token evaluation in ~10 s
  (reference perplexity 6.79 on the matched window). A full F16 evaluation would not fit, which
  is why the 8B reference is Q8_0.
* **`ModelSpec.alphabet`:** 8B genes are {INT8, INT4} only. The GA (`random_genome`, `mutate`),
  `MOSteadyStateGA` and baselines (`ModelSpec.baselines()`: uniform per precision + the
  heuristic mapped onto the alphabet) now honour it; tested that a search never proposes an excluded
  precision. Phi-3's baselines are unchanged.
* Bug found and fixed on the way: llama.cpp prints a `-nan` standard error when a candidate is
  identical to the base; the KLD parser choked on it (regression test added).

### 18.3 Llama-3.1-8B sensitivity (KLD vs the Q8_0 reference, Q4_0 only, 1023 tokens)

| block | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | sum | uniform |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4_0 vs Q8_0, 8B | 1.234 | 0.431 | 0.468 | 0.534 | 0.471 | 0.346 | 0.358 | 0.864 | 4.70 | 4.37 |
| Q4_0 vs F16, Phi-3 (for comparison) | 0.566 | 0.656 | 0.739 | 0.960 | 0.654 | 0.502 | 0.420 | 1.431 | 5.93 | 6.85 |

* 8B: both edges are the most sensitive (block 0 = 26% of the total, block 7 = 18%; edge/interior 2.4x), a
  U-shaped profile. Slightly sub-additive (sum 4.70 vs uniform 4.37), unlike Phi-3 (super-additive).
  Uniform Q4_0 costs 4.37% against the Q8_0 reference (Same-top-p 91.3%).
* **Pre-registered similarity check (plan D5) on the data so far, Phi-3 vs Llama-3.1-8B, Q4_0:**
  1. Edge blocks most sensitive in each model: **only partly** -- 8B yes (both edges); Phi-3 only the last
     block (block 0 is mid-ranked, 0.57%).
  2. Per-block profile rank correlation: **Spearman 0.55**, moderate, not high. The last block is
     top-2 in both; block 0 is top in 8B but 6th of 8 in Phi-3.
  Criteria 3-5 (Pareto fronts, crossover weighting, genome shape) need the searches (step 7 on) and the
  Arm measurements. Honest reading so far: the **"last block is sensitive" structure generalizes;
  "block 0 is sensitive" does not (it is architecture/model dependent)**. A third model (below) would
  separate "model-specific" from "size-specific".
* Caveat: the two tables differ in reference (F16 vs Q8_0) and size; the comparison is of per-block
  *shape*, which the Spearman/edge ratio are designed to capture, not of absolute cost.

### 18.4 Open for the user
* Accept the Llama-3.2-3B licence (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) so it can be
  downloaded; **or** substitute an ungated ~3B model (Qwen2.5-3B-Instruct, Apache-2.0: separate q/k/v with
  biases, 36 layers -> groups of 4 x 9 -> 8 uneven groups) -- plan D5 listed it as the alternative.
* Rotate the shared token when finished (it appeared in the conversation).

**Next:** see §19 (step 7 results).

## 19. T1 step 7: dry-run searches (real accuracy, simulated Arm speed/energy)

`scripts/t1_dry_run_search.py` runs the full Pareto GA with **measured accuracy** (KLD on the 3080,
GgufAssembler + LlamaKldProbe, `/dev/shm` work file) and a **simulated** speed/energy model driven by the
real per-block parameter counts, llama.cpp's real bits-per-weight (8.5 / 4.5) and the real output-head
bytes. Results are stamped `synthetic` (not Arm data): the point is to validate the pipeline and the
search before any Mac time is spent. Population 40, mutation 0.1, hypervolume stagnation stop (100).

| run | evaluations | distinct genomes assembled | wall | result |
|---|---|---|---|---|
| Phi-3-mini (3^8 = 6,561) | 600 (budget) | 414 (6.3% of space) | 23 min (2.3 s/eval) | front of 38, HV converged by ~eval 300 |
| Llama-3.1-8B (2^8 = 256) | 398 (stagnation) | 172 (67% of space) | 20 min (2.9 s/eval) | front of 34 (re-scored), HV converged by ~eval 100 |
Short checks first: 40 evals Phi-3 (159 s), 25 evals 8B (174 s); both fine.

### 19.1 Exhaustive accuracy table for the 8B model -> the true Pareto front
All 256 genomes were measured (`results/accuracy_table_llama3.1-8b.json`, ~7 s each, 30 min), and the
**true Pareto front over the whole space has only 9 points**, in a clean chain: starting from uniform Q8_0,
add Q4_0 to blocks in order of increasing sensitivity: 5, 6, 1, 2, 4, 3, 7, 0 -- *exactly* the ranking of the
single-block sensitivity table (§18.3).

| genome | 88888888 | 88888488 | 88888448 | 84888448 | 84488448 | 84484448 | 84444448 | 84444444 | 44444444 |
|---|---|---|---|---|---|---|---|---|---|
| accuracy cost % | 0.00 | 0.35 | 0.70 | 1.12 | 1.57 | 2.04 | 2.58 | 3.38 | 4.37 |
| simulated decode gain | 0.00 | +0.05 | +0.11 | +0.17 | +0.24 | +0.33 | +0.42 | +0.52 | +0.65 |

* **The GA found it:** hypervolume ratio **0.9999**, 8 of the 9 true points (recall 0.89), in 398 evaluations.
  Offline GA studies on the table (10 seeds each; capacity 20/40/80, 300-600 evaluations) all reach
  ratio 1.0000. **Caveat: this is a weak test**, since 300-600 evaluations of a 256-genome space is close
  to exhaustive. The meaningful test is the 6,561-genome Phi-3 space (see 19.3).
* **Honest consequence for the project's claim.** When the objective is additive across blocks (accuracy
  nearly additive as measured in §17/§18; bandwidth-bound decode time additive in bytes), the optimal front
  is *the greedy sensitivity ordering*: 17 measurements (the sensitivity table) give the same answer as
  ~400 GA evaluations. The GA is not needed for the 8B front in this model. Its value must be argued
  elsewhere: (a) it reaches the answer without the sensitivity table or the additivity assumption (shown
  here); (b) it can exploit **non-additive** structure -- Q4_0 on Phi-3 is 15% super-additive, and real Arm
  speed/energy may interact (cache effects, P/E-core scheduling, kernel tile shapes). **Plan change:** the
  final report includes a *sensitivity-greedy baseline* (k most sensitive blocks protected, k = 0..8) next to
  uniform and heuristic baselines and states whether the GA beats it on measured Arm data. `sensitivity_seeds`
  in the driver already builds these genomes (`--seed-sensitivity`).
* **Scalarization collapses (D2 confirmed, stronger than in the README):** every tested weighting --
  including accuracy weight 0.8 -- returns uniform Q4_0, because accuracy penalties (0-4%) are numerically tiny
  against speed gains (tens of %), so fitness is speed-dominated. The README needed w2 >= 0.8 to see a mixed
  genome on the 3080; here no tested weighting does. Operating points should therefore be chosen *from the
  Pareto front by an accuracy budget* ("fastest genome with <= x% perplexity increase"), not by weights.
  The scalarized GA run returns a single point (HV ratio 0.9888 against the 9-point front).
* The stagnation criterion fired at 398 evaluations because the population saturated the space; for 256
  genomes a smaller budget (or exhaustive search, 30 min) is the right tool.

### 19.2 Phi-3-mini front (6,561 genomes, 600 evaluations)
Front of 38 genomes (population 40, so the front nearly fills the population: a larger capacity would keep
more of it). Notable points (accuracy cost %, simulated decode gain): `88F88888` 0.058 / +0.59; `88888848` 0.47 /
+0.89; `44888888` 1.34 / +1.02; `44444444` 6.85 / +1.95. The published heuristic baseline `F884488F` (1.79% /
+0.64) is dominated, as in every earlier experiment. The true front is not known for this space (6,561 genomes).

### 19.3 Open: exhaustive Phi-3 table
Exhaustively measuring Phi-3 (6,561 genomes at ~3-4 s of new-assembly time each) is **~6 hours of unattended
GPU time** (`scripts/t1_exhaustive_accuracy.py --model phi3-mini`, resumable). It would give the true front, the
GA's real hypervolume ratio at 600 evaluations (9% of the space), a fair GA-vs-sensitivity-greedy comparison on
a space with genuine interactions (3 precisions, super-additive Q4_0), and exact offline GA studies
(`scripts/t1_front_quality.py`). Not started: it occupies the GPU and CPU for the duration.

### 19.4 Code
`tabulated.py` (exhaustive accuracy tables, `TabulatedAccuracyProbe`, `true_front`, `front_quality`),
`dryrun_setup.py`, `scripts/t1_dry_run_search.py`, `scripts/t1_exhaustive_accuracy.py`,
`scripts/t1_front_quality.py`; `SimulatedArmProbe` gained `bits` and `fixed_bytes`.
