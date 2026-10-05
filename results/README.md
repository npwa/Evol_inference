# results/ -- what is tracked and how it was produced

Raw measurements behind the claims in `Doc/implementation_plan_mac-m4.md` (the Arm port, branch `mac-m4`).
Every file is small and deterministic given its inputs; commands assume `PYTHONPATH=.`. Base-project run
snapshots (`phase6_*`, `phase7_*`) and resumable caches (`*_snapshot.json`) are not tracked (see `.gitignore`).

| File | What it is | Produced by | Plan section |
|---|---|---|---|
| `sensitivity_phi3-mini.json`, `sensitivity_phi3-mini_bnb.json` | per-block sensitivity from **perplexity deltas** (llama.cpp / bitsandbytes). Kept as evidence that this metric is noise-dominated (negative penalties) | `scripts/t1_sensitivity.py --metric ppl` | 17.1 |
| `sensitivity_phi3-mini_kld.json`, `..._bnb_kld.json`, `..._cpu_kld.json`, `sensitivity_llama3.1-8b_kld.json` | per-block sensitivity from **KL divergence** (CUDA, bitsandbytes, x86 CPU builds; Llama-3.1-8B) | `scripts/t1_sensitivity.py --metric kld [--build build-cpu] [--hf-bnb]` | 17.3, 18.3 |
| `t1_ppl_crosscheck.json` | perplexity of uniform genomes: bitsandbytes (two windows) vs llama.cpp CUDA vs CPU | `scripts/t1_crosscheck_ppl.py` | 16 (step 4) |
| `accuracy_table_llama3.1-8b.json` | measured KL divergence of all 256 genomes of the 8B model (RTX 3080, llama.cpp CUDA) | `scripts/t1_exhaustive_accuracy.py` | 19.1 |
| `dryrun_phi3-mini_pareto_s0.json`, `dryrun_llama3.1-8b_pareto_s0.json` | first dry-run Pareto searches: measured accuracy, **SIMULATED speed/energy with the original placeholder model** (`"synthetic": true`, `arm_model` absent = generic). Speed numbers superseded by the `_g3` files | `scripts/t1_dry_run_search.py` | 19 |
| `dryrun_phi3-mini_pareto_s0_g3.json`, `dryrun_llama3.1-8b_pareto_s0_g3.json` | the same searches with the **Graviton3-calibrated** speed model (`"arm_model": "graviton3"`); still synthetic, energy still a placeholder | `scripts/t1_dry_run_search.py --arm-model graviton3 --tag _g3` | 19.5 |
| `t2_arm_emulated.json` | cross-built aarch64 llama.cpp builds under QEMU vs x86 (KL divergence, kernel selection) | `scripts/t2_arm_emulated_check.py` | 20 |
| `t3_native_phi3-mini_<host>.json`, `t3_main.log` | AWS Graviton3 (c7g.2xlarge): KleidiAI on / `--no-repack` / no-KleidiAI build; accuracy and llama-bench speed | `scripts/t3_native_arm_check.py` | 22 |
| `t3_bw_probe.jsonl`, `t3_qdot_bench.jsonl`, `t3_qdot_test.txt` | Graviton3 memory-read bandwidth, own-kernel throughput, native bit-exactness test | `kernels/arm/{bw_probe,qdot_bench,qdot_test}` | 22 |
| `rehearsal_table_phi3-mini.jsonl` (+ `.meta.json`), `rehearsal_thread_scan.json`, `rehearsal_selftest.json`, `rehearsal_timings.json`, `rehearsal_analysis.json`, `rehearsal_queue.log`, `queue_rehearsal*` | the Graviton dress rehearsal of the Mac workflow (plan 23.2): 9 genomes x (stock, KleidiAI), per-evaluation timings, thread scan; mock energy meter (rows stamped synthetic) | `scripts/run_queue.py --init-rehearsal` | 23.2 |
| `rehearsal_attempt1_*` | the first, failed attempt (disk check on a RAM-backed /tmp): selftest JSON, log, queue state | same | 23.1 |
| `rehearsal_bw_probe.jsonl`, `rehearsal_qdot_bench.jsonl` | second Graviton3 instance: bandwidth and kernel throughput (reproduce the first within 2-3%) | `kernels/arm/*` | 23.1 |
| `t3_roofline.png` | decode bandwidth vs roof, and accuracy cost by KleidiAI configuration | `scripts/t3_report.py` | 22 |
| `m4_selftest.json` / `.log`, `m4_selftest_t10.json` / `.log` | Apple M4 (`mac-m4.metal`, macOS 26.7) selftest gate: platform, power response, per-configuration KL divergence and speed. `m4_selftest.json` is the queue's re-run at 8 threads, `_t10` the first manual run at 10 | `scripts/mac_selftest.py` | 23 |
| `m4_thread_scan.json` / `.log` | M4 thread scan (1-10 threads, stock and KleidiAI, Phi-3 uniform Q8_0 / Q4_0) with the real powermetrics meter: decode, prefill, joules per token | `scripts/m4_thread_scan.py` | 23 |
| `m4_bootstrap.json` | the M4 bootstrap record (Homebrew, venv, two llama.cpp builds, models, powermetrics checks) | `scripts/bootstrap_mac.py` | 23 |
| `m4_table_llama3.1-8b.jsonl`, `m4_table_phi3-mini.jsonl` (+ `.meta.json`) | the Apple M4 measured tables, one JSON line per (genome, configuration): KL divergence, decode / prefill speed, joules per token (real powermetrics meter). 8B: 256 genomes `kai` + 12 `stock`; Phi-3: 256 genomes in both configurations plus the F16 reference and a few extra F16 rows (use `--alphabet int8 int4`) | `scripts/m4_measure_table.py` | 24 |
| `m4_analysis_llama3.1-8b.json`, `m4_analysis_phi3-mini.json` | offline analysis of those tables: fronts, GA and greedy studies, `kai` vs `stock` | `scripts/m4_analyze.py [--alphabet int8 int4]` | 24 |
| `queue_m4.json`, `queue_m4.state.json`, `m4_queue.log`, `m4_phi3_stock_extension.log` | the unattended queue (spec, state, log) and the follow-up run that measured `stock` for the rest of the Phi-3 space | `scripts/run_queue.py --init-m4` | 24 |

**Provenance rules.** Result JSONs carry a `platform` block (host, kernel, Arm features, llama.cpp commit); report
code refuses files without one. Anything produced with a simulated probe is stamped `synthetic` and must not be quoted
as an Arm measurement. Accuracy numbers are only comparable within one scored window (`llama-perplexity` scores the
second half of each 2048-token chunk) and one machine/kernel path (the same Q8_0 bytes give different accuracy on
CUDA, x86 and under KleidiAI).

The raw M4 captures used by the parser tests (`powermetrics` plist, `pmset`, `sysctl`) are in `tests/fixtures/m4_real/`.
