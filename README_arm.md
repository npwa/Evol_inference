# Evol_inference on Arm (branch `mac-m4`)

Re-targets the mixed-precision quantization search from "bitsandbytes on an RTX 3080" to **LLM inference on Arm
CPUs** (AWS Graviton, then an Apple M4 bare-metal Mac), with **three objectives: accuracy, decode speed and
energy per token**, searched as a Pareto front. The unifying lesson of the original project carries over and
gets sharper: *the cost model has to be measured on the target silicon.* Full decision log and every number
below: [`Doc/implementation_plan_mac-m4.md`](Doc/implementation_plan_mac-m4.md). Slides:
`presentation/Evol_inference_mac-m4.pptx`. Raw measurements: [`results/README.md`](results/README.md).

## Status

| Tier | Where | Cost | Status |
|---|---|---|---|
| T0 | desktop, no GPU: unit tests, GA, GGUF assembler, parsers, energy-meter backends | free | **done** |
| T1 | desktop + RTX 3080: real models, real accuracy, 2 search models (Phi-3-mini, Llama-3.1-8B), dry-run searches | free | **done** (Llama-3.2-3B blocked on licence access) |
| T2 | aarch64 emulation (QEMU): cross-built llama.cpp and NEON kernels, correctness only | free | **done** |
| T3 | AWS Graviton3 (`c7g.2xlarge`): KleidiAI on/off, roofline, kernels on real hardware | about 2 USD | **done** ([runbook](Doc/graviton_runbook.md)) |
| T4 | AWS `mac-m4.metal` (Apple M4, 10 cores, 24 GiB): the only source of M4 speed and energy | about 30 USD (1.23 USD/h, 24 h minimum), runbook: [`Doc/m4_runbook.md`](Doc/m4_runbook.md) | **in progress**: selftest and thread scan done (SME2 works; KleidiAI Q8_0 loss reproduces, no decode gain), measurement tables running |

## What was found (all measured unless marked simulated)

* **Perplexity deltas cannot resolve per-block sensitivity** on ~1000-2000 tokens (negative penalties, uncorrelated
  profiles across backends). Accuracy is therefore **KL divergence to the original model's logits**
  (`accuracy_penalty = exp(KLD) - 1`, about 5% relative error, 1e-5 nats noise floor). This also *revises* the
  GPU project's per-block claim: block 0 carries ~33% of bitsandbytes INT8's cost (3-5x an interior block), not
  70% / 30x.
* **4-bit sensitivity is quantizer-independent** (Q4_0 vs NF4 per-block profiles: Spearman 0.976, last block most
  sensitive); **8-bit is not** (0.35). Q8_0 is almost exactly additive across blocks. Across models the structure
  generalizes only partly (Phi-3 vs Llama-3.1-8B Q4_0: Spearman 0.55).
* **The search recovers the exact Pareto front** (exhaustive 8B table, 256 genomes: hypervolume ratio 0.9999), **but so
  does a greedy rule from the sensitivity table** (17 measurements) when objectives are additive. Scalarized
  weights collapse to uniform Q4_0 for every weighting tried; choose operating points from the front by accuracy
  budget. *The speed and energy in these searches are simulated.*
* **KleidiAI's Q8_0 path re-quantizes each weight row to one scale and is 37x less accurate on Phi-3** (KLD 0.0283 vs
  0.00076; 8x on a 360M proxy), with **no decode speedup over llama.cpp's own Arm path** on Graviton3 (0.94-1.05x;
  Q8_0 prefill +1.34-1.47x; Q4_0 identical). Against `--no-repack` it looks 2.3x faster, which is the wrong baseline.
  This does not predict the M4 (KleidiAI uses SME2 there) and must be measured.
* **Decode is bandwidth-bound and smaller is faster on Arm** (unlike the GPU): Q4_0 decodes 1.41-1.52x faster than Q8_0.
  Q8_0 decode sits at 95-99% of the measured memory roof, Q4_0 at 67-80%, F16 at 37-48% (no optimized kernel; a poor
  baseline).
* **Own NEON kernels** (`kernels/arm/qdot`, Q8_0/Q4_0 dot products with SDOT and SMMLA) are bit-identical to the
  reference on real Graviton3 and under emulation (108 cases; a deliberate-bug check is caught 1,710 times), but
  reach only 47% / 31% of llama.cpp's single-thread bandwidth. Correct, not yet fast.
* KleidiAI's **SME kernels give garbage under QEMU 8.2.2**: cannot be attributed to the emulator vs the kernel, so
  it is a selftest gate for the Mac run.

## Layout (Arm port)

| Path | Purpose |
|---|---|
| `evol_inference/genome.py`, `model_spec.py` | bitsandbytes-free genome types; per-model grouping, tensor names, precision alphabet (Phi-3-mini, Llama-3.1-8B, Llama-3.2-3B, SmolLM2-360M proxy) |
| `evol_inference/gguf_assembler.py` | assembles a genome into one GGUF by splicing pre-quantized tensors (bit-identical to `llama-quantize --tensor-type`) |
| `evol_inference/llama_probes.py` | `llama-perplexity` (perplexity, KL divergence) and `llama-bench` probes; parsers for `system_info` and KleidiAI kernel selection |
| `evol_inference/objectives.py`, `mo_fitness.py`, `mo_ga.py`, `pareto_nd.py` | three-objective evaluator, steady-state GA with Pareto or scalar ranking, N-D Pareto math |
| `evol_inference/sensitivity.py`, `tabulated.py` | per-block sensitivity tables; exhaustive accuracy tables, true front, front quality |
| `evol_inference/probes.py`, `platform_info.py`, `dryrun_setup.py` | simulated probes (incl. Graviton3-calibrated), platform record, shared setup |
| `energy_meter.py` | energy measurement: `powermetrics` (Mac), RAPL (x86), mock, replay |
| `evol_inference/mac_measure.py`, `measured_table.py`, `mac_env.py`, `selftest_gates.py`, `job_queue.py` | Mac-day tooling: per-genome measurement under several run configurations, measured-table probes, thermal state, selftest gate logic, unattended queue |
| `scripts/mac_selftest.py`, `m4_measure_table.py`, `m4_thread_scan.py`, `run_queue.py`, `m4_analyze.py`, `m4_timing_report.py`, `bootstrap_mac.py` | the T4 workflow: gate, measure every genome, thread scan, queue within a budget, offline analysis, timing projections, bootstrap (dry-run first) |
| `kernels/arm/` | `qdot` NEON kernels + tests + benchmark, `bw_probe`, `hwcap_probe`, aarch64 toolchain file |
| `scripts/t1_*.py` | tier T1 (sensitivity, cross-checks, dry-run searches, exhaustive table, front quality) |
| `scripts/t2_arm_emulated_check.py` | tier T2 (aarch64 builds under QEMU vs x86) |
| `scripts/t3_native_arm_check.py`, `t3_report.py` | tier T3 (native Arm: KleidiAI on/off accuracy and speed; tables and roofline figure) |
| `Doc/implementation_plan_mac-m4.md`, `Doc/graviton_runbook.md` | plan, decisions and results; exact AWS steps |

## Running things

```bash
PYTHONPATH=. python -m pytest -q -m "not gpu and not llamacpp and not slow"   # any host, no GPU, no bitsandbytes needed
PYTHONPATH=. python -m pytest -q -m "llamacpp"                                # needs ~/work/llama.cpp builds + GGUFs (T1)
PYTHONPATH=. python scripts/mo_search_synthetic.py --model llama3.2-3b        # synthetic end-to-end search (T0)
PYTHONPATH=. python scripts/t3_report.py                                      # regenerate the T3 tables and figure
```
Markers: `gpu`, `llamacpp`, `arm_emulated`, `arm_native`, `mac_only`, `slow`. Building llama.cpp (CPU, CUDA,
cross-compiled aarch64) and converting models are described in plan sections 16 and 20; the Graviton run in
`Doc/graviton_runbook.md`.

## Energy measurement (`energy_meter.py`)

One interface, `EnergyMeter`, with `start()`, `stop()`, `idle_baseline(seconds)` and `measure_cmd(cmd)`; each
measurement is a `Measurement` (`duration_s`, `energy_j`, `avg_power_w`, `net_energy_j(baseline_w)`). Backend by
`ENERGY_BACKEND`, auto-detected otherwise:

* `powermetrics`: macOS on Apple silicon, streams `powermetrics -f plist` (NUL-separated samples), CPU power.
  Needs passwordless `sudo -n /usr/bin/powermetrics`.
* `rapl`: Linux x86 via `/sys/class/powercap/intel-rapl:0` (root-readable only by default; the real-RAPL test skips otherwise).
* `mock`: synthetic power from CPU time used plus noise. Results measured with it are stamped synthetic.
* `replay`: a recorded powermetrics fixture run through the real parser (parser tests only).

The parser, baseline subtraction, RAPL wrap-around, the streaming loop (against a fake `powermetrics` process) and
backend selection are tested (`tests/test_energy_meter.py`). **Open risk, closed only on the Mac:** the plist key
names (`cpu_power` in mW, `cpu_energy` fallback) are unverified. First action on the Mac:
`python energy_meter.py record pm_fixture.plist --seconds 10` (prints the real keys if none match), then
`selftest`, and commit the fixture so `replay` tests the real format on Linux. Graviton exposes no energy counter,
so T3 has no power data.

## Rules that keep the Mac day cheap

* Write harness logic in Python, not bash (macOS lacks GNU `timeout`, `taskset`, `perf`; `sed`/`date` differ);
  control thread **count** only (no pinning on macOS).
* Build llama.cpp with `-DGGML_CPU_KLEIDIAI=ON`; on the Mac also `-DGGML_METAL=OFF` / `-ngl 0` so the CPU path runs.
  Keep a **second build without KleidiAI** as the speed baseline; never compare against `--no-repack`.
* Linux (x86, Graviton) validates correctness and accuracy; **never use their latency or energy as M4 fitness**. The
  Graviton run is used to calibrate the simulated model and to find problems, not as M4 data.
* Models up to ~8B parameters (M4: 24 GiB unified memory); for Llama-3.1-8B the reference is Q8_0 (F16 does not fit).
* Before any search, run the **selftest gates** (plan sections 20.4 and 22.6): Q8_0/Q4_0 KL divergence with KleidiAI
  default, with SME disabled (`GGML_KLEIDIAI_SME=0`) and without KleidiAI; abort if the default path is more than 200x
  worse than stock (garbage). Record KleidiAI's kernel selection (`-v`) with every result.
* Bootstrap (`bootstrap_mac.sh`, thin; logic in Python), pre-stage models, queue unattended runs with snapshots every
  N evaluations, and a budget guard so the 24 h block is filled with seeds and models rather than idle.

## Limits stated plainly

Speed and energy in all searches so far are **simulated** (calibrated to Graviton3, not the M4); SME2 is untested
(QEMU 8.2.2 lacks it); no power measurement exists yet; Llama-3.2-3B awaits licence access; the Graviton run did not
include `perf` counters or Graviton4.
