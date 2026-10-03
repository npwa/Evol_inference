# Graviton dress rehearsal of the Mac workflow

Purpose: run the **same pipeline the Mac day will run** (selftest gate, thread scan, table measurement, offline
analysis) on real Arm Linux for about one dollar, to (1) replace the placeholder time estimates in the default Mac
queue with **measured per-evaluation costs**, which decide how many genomes fit in the 24 h block (about 40 USD),
and (2) catch Arm-specific bugs in the new tooling before that block starts (the desktop rehearsal already found one).

Instance, image, setup and teardown are the T3 steps in [`graviton_runbook.md`](graviton_runbook.md); this file lists
only what is different. Background: `implementation_plan_mac-m4.md` §23.

**What it cannot check:** Graviton has no energy counter, so the rehearsal uses the **mock** meter (every row is stamped
`synthetic`); `powermetrics`, `pmset`, the macOS bootstrap and the Metal-off check remain untested until the Mac.

## Time and cost

About **2.5 hours of wall time** on a `c7g.2xlarge` (under 1 USD on-demand), of which about 70 minutes is the rehearsal
itself and 60-75 minutes is first-time setup. Taking an image after setup (step 2) makes any repeat skip the setup.
Estimates are scaled from the T3 Phi-3 speeds, accurate to about +/-30%:

| Step | Time |
|---|---|
| Setup (runbook steps 1-8): packages, clone, venv, two builds, Phi-3 GGUFs | 60-75 min (first time only) |
| `selftest`: 2 configurations (stock, KleidiAI), context 512; the F16 base logits dominate | ~8 min |
| `thread-scan`: Q8_0 and Q4_0 at 1 and 8 threads, both configurations | ~13 min |
| `table-phi3-mini`: F16 reference (about 6.7 min per configuration) + 11 quantized genomes x 2 configurations (about 80 s each) + 3 min base logits | ~45 min |
| `analyze` + `timings` | ~2 min |

Shorter: `--limit 6` (about 30 min for the table) or delete the `thread-scan` job from the queue file.

## 1. Setup differences from the runbook

Follow runbook steps 1-8 with these changes:

* the clone must contain the Mac tooling: it is in commit `4b8184b` and later; **push `mac-m4` first**, then `git pull` on the
  instance (step 5);
* the build without KleidiAI is named **`build-nokai`** there (step 6); the rehearsal queue uses that name by default;
* Phi-3 only: skip the Llama downloads; step 7 (kernels) and the step 9 tests are optional but cheap; run the tests once:
  ```bash
  cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/llama.cpp
  python -m pytest -q -m "not gpu and not llamacpp and not slow"        # expect all to pass, a few skipped
  git log -1 --format='%H %s' | tee ~/rehearsal_commit.txt
  ```

## 2. Optional: save an image of the prepared instance **[desktop]**

So the next rehearsal (or a re-run after a fix) skips the hour of setup. Stop being charged for it by deleting it when done
(an image of the 64 GiB volume costs about 2-3 USD per month in snapshot storage).
```bash
aws ec2 create-image --region REGION --instance-id i-XXXXXXXX --name evol-graviton-ready --description "Phi-3 GGUFs, llama.cpp builds, venv"
# later: launch from the returned ami-... exactly like runbook step 3, and `git pull` to get newer tooling.
# cleanup when finished: aws ec2 deregister-image --image-id ami-... ; then delete its snapshot in the console (EC2 > Snapshots).
```

## 3. Run the rehearsal queue

```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/llama.cpp
python scripts/run_queue.py --init-rehearsal queue_rehearsal.json --limit 12 --threads 8
cat queue_rehearsal.json | head -30                    # check it: five jobs, selftest first and required
nohup python -u scripts/run_queue.py queue_rehearsal.json > results/rehearsal_queue.log 2>&1 &
tail -f results/rehearsal_queue.log
```
* Per-job logs: `results/queue_logs/<job>.log` (for example `tail -f results/queue_logs/table-phi3-mini.log` shows each genome as
  it is measured, with KL divergence, decode speed and the time per genome).
* The queue's budget is 3.5 h; it skips a job that cannot finish and gives the table job the time that is left.
* **Stop cleanly** after the current job: `touch queue_rehearsal.state.STOP`. **Resume** after any interruption by running the
  same `run_queue.py` command again: finished jobs are skipped, the table driver continues where it stopped.
* Do not run anything else CPU-heavy on the instance meanwhile: it distorts the timings you are here to measure.

## 4. What success looks like

* `results/rehearsal_queue.log` ends with `queue finished`, every job `done`.
* **Selftest** (`results/rehearsal_selftest.json`): `"ok": true`, `usable_configs` = `["stock", "kai"]`; the `q8-sane[kai]` check
  reports the expected KleidiAI loss (T3 saw 37x on Phi-3 at context 2048; at context 512 the factor may differ, anything between
  about 3x and 200x is the "expected" band); `q4-agrees[kai]` within 10%; `cpu-only[...]` ok; `energy-responds-to-load` ok
  with `mock` (a synthetic backend never fails the gate).
* **Table** (`results/rehearsal_table_phi3-mini.jsonl`): 12 genomes x 2 configurations, **no `"error"` rows**; the first genome is the F16
  reference, then uniform Q8_0, uniform Q4_0, then the greedy chain.
* **Timings** (`results/rehearsal_timings.json`, also printed at the end of `results/queue_logs/timings.log`): median seconds per
  (genome, configuration) split into the KL run and the speed run, F16-containing genomes reported separately, and projections of
  the 2-level space (256 genomes) for Phi-3 and Llama-3.1-8B for a Mac that is 1x, 1.5x, 2x, 3x faster than this instance.
  **This table is the deliverable**: it replaces the placeholder estimates in the default Mac queue.

Anything else (a failed job, a crash, an `"error"` row, a gate that fails unexpectedly) is a **finding**, not a nuisance: keep the log and
send it; it is exactly what the rehearsal is for.

## 5. Bring results back and shut down

```bash
# [desktop]
rsync -avz -e "ssh -i ~/.ssh/YOUR_KEY.pem" ubuntu@PUBIP:~/Evol_inference/results/ ~/work/Evol_inference/results/
rsync -avz -e "ssh -i ~/.ssh/YOUR_KEY.pem" ubuntu@PUBIP:~/Evol_inference/queue_rehearsal.* ~/work/Evol_inference/results/
# [desktop] then terminate the instance (billing stops); see runbook step 13
aws ec2 terminate-instances --region REGION --instance-ids i-XXXXXXXX
```
The `results/rehearsal_*` and `results/queue_rehearsal*` files are small and already excepted in `.gitignore`, so once reviewed they can be
committed like the other measured results (the per-job logs in `results/queue_logs/` stay ignored).

## 6. Optional extension: rehearse the 8B model too (adds about 1.5 h and 0.5 USD)

The projections scale Phi-3's per-evaluation cost to Llama-3.1-8B by a factor of **2.1** (weight bytes), an estimate. Measuring
it removes that assumption. It needs a 100 GiB disk, a Hugging Face token with access to the model, and the same conversion steps
as runbook step 8 for `meta-llama/Llama-3.1-8B-Instruct` (F16 16 GB; make `llama3.1-8b-q8_0-pure.gguf` and
`llama3.1-8b-q4_0-pure.gguf` with `llama-quantize --pure`). Then:
```bash
python scripts/run_queue.py --init-rehearsal queue_rehearsal_8b.json --model llama3.1-8b --limit 8 --threads 8
nohup python -u scripts/run_queue.py queue_rehearsal_8b.json > results/rehearsal_queue_8b.log 2>&1 &
```
(the 8B alphabet is already Q8_0/Q4_0 and its reference is Q8_0, so no F16 reference run is needed).

## 7. If something goes wrong

| Symptom | Likely cause / action |
|---|---|
| `selftest` FAILED, `files-present` | a build or GGUF name differs: check `ls ~/llama.cpp/build-kai/bin build-nokai/bin` and `models/gguf/`; the queue assumes `build-kai`, `build-nokai`, `phi3-mini-{f16,q8_0-pure,q4_0-pure}.gguf` |
| `selftest` FAILED, `q8-sane[kai]` | the KleidiAI Q8_0 error is >200x stock: a real finding on Arm Linux (keep the JSON), the queue aborts by design; rerun with `--configs stock kai-nr` to continue without it |
| `cpu-only` fails | the build has a GPU backend: rebuild without it (the Graviton builds in the runbook are CPU only) |
| a table row has `"error"` | the message is in the row; the genome is retried on resume; send the row |
| `memory` check fails | the instance is too small (the F16 reference needs about 9 GB resident; use 16 GiB) |
| the queue skips `table-...` (`skipped_budget`) | the earlier jobs overran the 3.5 h budget: rerun with a bigger `--budget-hours` in `--init-rehearsal` |
