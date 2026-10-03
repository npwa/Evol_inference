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
| `t3_roofline.png` | decode bandwidth vs roof, and accuracy cost by KleidiAI configuration | `scripts/t3_report.py` | 22 |

**Provenance rules.** Result JSONs carry a `platform` block (host, kernel, Arm features, llama.cpp commit); report
code refuses files without one. Anything produced with a simulated probe is stamped `synthetic` and must not be quoted
as an Arm measurement. Accuracy numbers are only comparable within one scored window (`llama-perplexity` scores the
second half of each 2048-token chunk) and one machine/kernel path (the same Q8_0 bytes give different accuracy on
CUDA, x86 and under KleidiAI).
