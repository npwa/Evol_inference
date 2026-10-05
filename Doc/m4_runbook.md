# Runbook: tier T4, the Apple M4 measurement day (AWS `mac-m4.metal`)

**Status: v0, written without Mac access.** Every AWS command below follows the AWS documentation; every macOS-specific step
(`powermetrics`, `pmset`, Homebrew, the bootstrap, Metal-off builds) has been **tested only as generated plans and parsers on
Linux**. Section 4 lists the checks for the first minutes, which will confirm or correct them. Update this file from what actually
happens. Background and the reasoning behind the design: `implementation_plan_mac-m4.md` §20, §22, §23.

## 0. What the day does, and what it costs

A **Dedicated Host** is allocated and billed per second with a **24-hour minimum**, so the whole day is sunk cost once you allocate
it: use it. About **1.23 USD per hour** on-demand (price from a third-party list; check the AWS pricing page), so about **30 USD**
for the minimum, plus about 1 USD for the 200 GiB disk and S3. The host keeps billing until you **release** it, which is only
allowed after the 24 hours and after the instance is terminated (the host is then scrubbed, which can take **up to 4.5 hours** on
Apple silicon). Release it promptly (section 9).

The experiment (decisions D1-D5, plan §14): for each of two models, **measure every genome once** (256 over Q8_0/Q4_0) in two run
configurations, **`stock`** (llama.cpp built without KleidiAI: the fair baseline) and **`kai`** (KleidiAI, SME2 on the M4),
recording accuracy (KL divergence vs the original model), decode and prefill speed, and energy per token from `powermetrics`;
then analyse offline on the desktop (true front, GA and sensitivity-greedy against it, KleidiAI vs stock).

Timeline (hours after allocating the host; the clock is the 24 h minimum):

| T+ | What |
|---|---|
| 0:00-0:20 | allocate host, launch instance (macOS instances take 10-15 min to become reachable) |
| 0:20-1:00 | first checks (section 4), bootstrap and the two llama.cpp builds (section 5) |
| 1:00-1:30 | models from S3, checksums (section 6) |
| 1:30-2:30 | selftest gate (section 7) and, if it fails, diagnosis: **leave at least 1.5 h of slack** |
| 2:30-3:15 | thread scan, choose the thread count (section 8) |
| 3:15-21:45 | the queue (section 9), budget about 18 h |
| 21:45-24:00 | analysis check, results to S3 and desktop, terminate instance (section 10) |
| 24:00+ | release the host as soon as AWS allows it (section 11) |

## 1. Before the day (days ahead; some of these take days)

1. **Service quota.** The default quota for Mac Dedicated Hosts is **0**. In the console: Service Quotas > Amazon EC2 > search
   "mac-m4" (the exact quota name is "Running Dedicated mac-m4 Hosts"; check) > request 1. **Do this first**, approval can take days.
2. **Region and zone.** `mac-m4.metal` is offered in a limited set of regions (us-east-1, us-east-2, us-west-2, eu-central-1 among
   them). Check your zones:
   ```bash
   aws ec2 describe-instance-type-offerings --region us-east-1 --location-type availability-zone \
     --filters Name=instance-type,Values=mac-m4.metal --query 'InstanceTypeOfferings[].Location' --output text
   ```
3. **S3 bucket in that region** (`BUCKET`), and an **instance role** so the Mac can read models and write results without keys:
   ```bash
   cat > trust.json <<'JSON'
   {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
   JSON
   cat > s3policy.json <<'JSON'
   {"Version":"2012-10-17","Statement":[
    {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::BUCKET"},
    {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::BUCKET/models/*"},
    {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject"],"Resource":"arn:aws:s3:::BUCKET/run1/*"}]}
   JSON
   aws iam create-role --role-name evol-mac --assume-role-policy-document file://trust.json
   aws iam put-role-policy --role-name evol-mac --policy-name s3 --policy-document file://s3policy.json
   aws iam create-instance-profile --instance-profile-name evol-mac
   aws iam add-role-to-instance-profile --instance-profile-name evol-mac --role-name evol-mac
   ```
4. **Stage the models in S3** (about 27 GB; the files are listed with their checksums in `Doc/model_checksums.sha256`). From the desktop
   (resumable; time = 27 GB / your upload speed, which can be hours: start days ahead):
   ```bash
   cd ~/work/Evol_inference/models/gguf
   aws s3 cp wikitext2_test.txt s3://BUCKET/models/
   for f in phi3-mini-f16 phi3-mini-q8_0-pure phi3-mini-q4_0-pure llama3.1-8b-q8_0-pure llama3.1-8b-q4_0-pure; do
     aws s3 cp $f.gguf s3://BUCKET/models/; done
   ```
   (No Llama F16 is needed: the 8B reference is Q8_0. Alternative if your upload is slow: produce the GGUFs on an AWS instance as in the
   Graviton runbook step 8 and copy them to S3 from there at AWS speed; their file hashes will differ from the desktop's, so record which
   set you used.)
5. **Everything pushed:** `git push origin mac-m4` with the tooling, this runbook and the sensitivity tables (they are tracked in
   `results/` and the table driver uses them for ordering).
6. **The Graviton rehearsal done** (`graviton_rehearsal.md`): its `results/rehearsal_timings.json` tells you the per-evaluation cost and
   therefore how to split the queue between the two models (section 9). Without it the split is a guess.
7. **AWS Budgets alarm** (about 45 USD) so a forgotten host cannot run up a bill.

## 2. Allocate the host and launch the instance **[desktop]**

```bash
export R=us-east-1 AZ=us-east-1b KEY=YOUR_KEY            # your region, zone and key pair
# a security group allowing SSH from your IP only (as in graviton_runbook.md step 3), then:
HOST=$(aws ec2 allocate-hosts --region $R --availability-zone $AZ --instance-type mac-m4.metal --quantity 1 \
        --auto-placement off --tag-specifications 'ResourceType=dedicated-host,Tags=[{Key=Name,Value=evol-m4}]' \
        --query 'HostIds[0]' --output text); echo $HOST
# note the time: the 24 h minimum starts now. "Insufficient capacity" -> try another zone from section 1.2.

# find the macOS AMI (Apple-silicon, macOS 15.6 or newer is required for M4); list what exists, then pick:
aws ssm get-parameters-by-path --region $R --path /aws/service/ec2-macos --recursive \
  --query "Parameters[?contains(Name,'arm64_mac') && contains(Name,'image_id')].Name" --output text
AMI=$(aws ssm get-parameter --region $R --name /aws/service/ec2-macos/tahoe/arm64_mac/latest/image_id --query Parameter.Value --output text)
# expected image: macOS Tahoe 26.7, ami-0d2a9e55f71bda474 (built by AWS 2026-09-22, user ec2-user; includes AWS CLI, Xcode Command Line
# Tools, SSM agent and Homebrew, so the bootstrap should find them already installed). `echo $AMI` should match or be newer; AMI IDs are per region.
# (chosen: macOS 26 Tahoe, a year old; macOS 27 was released 2026-09-14 and is avoided on purpose: toolchain and powermetrics
#  format are least settled on a brand-new major release. If the listing shows no 'tahoe' path, use the newest 26.x one. Do not run softwareupdate.)

aws ec2 run-instances --region $R --instance-type mac-m4.metal --image-id $AMI --key-name $KEY \
  --security-group-ids sg-XXXXXXXX --iam-instance-profile Name=evol-mac --placement "Tenancy=host,HostId=$HOST" \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,Iops=6000,Throughput=500,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=evol-m4}]' --metadata-options HttpTokens=required
aws ec2 wait instance-status-ok --region $R --instance-ids i-XXXXXXXX      # 10-15 minutes is normal
ssh -A -i ~/.ssh/$KEY.pem ec2-user@PUBLIC_IP                                # the macOS user is ec2-user; -A forwards your GitHub key
```

## 3. Get the code

```bash
sw_vers && uname -m                                  # expect macOS 15.6+ and arm64
git clone --branch mac-m4 --single-branch git@github.com:npwa/Evol_inference.git ~/Evol_inference      # private repo via the forwarded agent
#  public repo: git clone --branch mac-m4 --single-branch https://github.com/npwa/Evol_inference.git ~/Evol_inference
cd ~/Evol_inference && git log -1 --format='%H %s' | tee ~/m4_commit.txt
mkdir -p results
```
If `git` is missing, the Xcode command line tools are not installed: run `xcode-select --install` (interactive; on a headless instance use
`softwareupdate` to install the "Command Line Tools" package) and retry.

## 4. First checks (about 15 minutes) that confirm or correct what was assumed

Do these **before** the bootstrap, saving every raw output (they become test fixtures). Anything that differs from the expectation is a
finding: fix the cause (or the code) now, while there is time.

```bash
cd ~/Evol_inference
sudo -n true && echo "passwordless sudo: ok"                                           # expected ok on EC2 Mac AMIs
sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu | tee results/m4_sysctl.txt
sysctl -a | grep -E "hw.optional.arm|hw.optional.AdvSIMD" >> results/m4_sysctl.txt       # FEAT_SME, FEAT_SME2, FEAT_I8MM ...
pmset -g therm | tee results/m4_pmset_therm.txt ; pmset -g | tee results/m4_pmset.txt  # raw text: fixtures for evol_inference/mac_env.py
sudo pmset -a lowpowermode 0 ; sudo pmset -a sleep 0 disksleep 0 displaysleep 0        # no Low Power Mode, never sleep
sudo mdutil -a -i off                                                                  # stop Spotlight indexing the large GGUFs (CPU noise)
sudo powermetrics --samplers cpu_power -i 500 -n 3 -f plist | head -c 3000 | tee results/m4_powermetrics_sample.txt
```
Expected: 10 CPUs, 24 GiB, **4 performance + 6 efficiency** cores; `FEAT_SME2` present; `lowpowermode 0`; `powermetrics` prints a plist.
**Record the key names** inside its `processor` dictionary: the parser assumes `cpu_power` (milliwatts) with `combined_power` and
`cpu_energy` as fallbacks. If none exists, update `POWER_KEYS_MW` in `energy_meter.py` to the real key before continuing.

## 5. Bootstrap (about 40 minutes: mostly the two builds)

```bash
cd ~/Evol_inference
python3 scripts/bootstrap_mac.py | tee ~/bootstrap_plan.txt      # DRY RUN: read it; it prints every command (runs under the macOS system Python 3.9: the script is written for that)
python3 scripts/bootstrap_mac.py --execute --llama-dir ~/work/llama.cpp --repo ~/Evol_inference --s3-models s3://BUCKET/models --jobs 10
```
The plan installs Homebrew (if absent), `cmake git python@3.12 awscli`, a Python 3.12 virtual environment, clones llama.cpp at the pinned
commit `ec7630a`, builds **`build-kai`** (KleidiAI on) and **`build-stock`** (off), both with **Metal, Accelerate, BLAS and OpenMP off**
(CPU only; the stock build is then llama.cpp's own NEON/I8MM path rather than Apple's BLAS), syncs the models from S3, then checks
`sudo -n powermetrics`, switches Low Power Mode off and **records the powermetrics fixture** (`pm_fixture.plist`). It stops at the first
failure and writes `results/m4_bootstrap.json`. Failures here are expected to need a fix: the steps were never run on a Mac.
Notes: a RAM disk is deliberately **not** used (24 GiB total memory; an SSD-backed work dir is fast enough and safer); add
`--with-accelerate-build` only if you want Apple's Accelerate BLAS as a third baseline (costs another build and another configuration).

After it succeeds: `git add pm_fixture.plist` later (commit it from the desktop with the results) and
`. .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/work/llama.cpp`.

## 6. Models and checksums

```bash
cd ~/Evol_inference/models/gguf && ls -la
shasum -a 256 -c ~/Evol_inference/Doc/model_checksums.sha256      # all six must say OK (macOS uses shasum, not sha256sum)
```
A mismatch means the wrong files or a corrupted transfer: fix it now. (If you generated the GGUFs on a cloud instance instead of copying
the desktop's, the F16 and derived hashes differ; compare tensor digests as described in the Graviton discussion, and record which set ran.)

## 7. The selftest gate (do not skip; run it by hand first, not inside the queue)

```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/work/llama.cpp
python scripts/mac_selftest.py --model phi3-mini --configs stock kai kai-nosme --meter powermetrics --threads 10 \
  --out results/m4_selftest.json 2>&1 | tee results/m4_selftest.log
```
(About 10 minutes.) It checks: macOS arm64, disk and memory, Low Power Mode off, thermal state, **that powermetrics power rises under
load**, that `pm_fixture.plist` exists, **llama.cpp runs CPU-only (no Metal)**, and for each configuration the KL divergence of
F16 / Q8_0 / Q4_0 against the stock path, plus decode speed. It prints which configurations are usable and writes
`results/m4_selftest.json`. How to read the outcomes (from the Graviton results, `implementation_plan_mac-m4.md` §22):

| Result | Meaning / action |
|---|---|
| `q8-sane[kai]` "EXPECTED KleidiAI loss" (3-200x) | normal: KleidiAI re-quantizes Q8_0 weights per row (Graviton: 28-37x). Recorded, continue. |
| `q8-sane[kai]` "WRONG KERNEL" (>200x), `kai-nosme` fine | the SME path produces wrong output (as under QEMU). `kai` is dropped automatically; continue with `stock` and `kai-nosme` (edit the queue's `--configs`), and **keep the JSON: it is a finding for the KleidiAI maintainers** |
| `q8-sane[stock]` fails | something is wrong with the build or the files: stop and diagnose before spending more of the day |
| `cpu-only` fails | a GPU backend is active: rebuild with `-DGGML_METAL=OFF` (the bootstrap does) |
| `energy-responds-to-load` fails | powermetrics returns no usable samples or flat power: fix the parser keys or sudo (section 4) |
| `thermal` warning | the machine reports throttling: wait and rerun (`pmset -g therm`) |
| `low-power-mode` fails | `sudo pmset -a lowpowermode 0` |
| `disk` / `memory` fails | the message names the location and the number |

Exit status 0 means it may proceed. **Budget slack for this step is deliberate**: it is the only protection for the other 20 hours.

## 8. The thread scan, and choosing the thread count (about 30 minutes)

macOS cannot pin threads, so only the **count** is a knob; with 4 performance and 6 efficiency cores the count decides which cores run.
```bash
python scripts/m4_thread_scan.py --model phi3-mini --configs stock kai --threads 1 2 4 6 8 10 --meter powermetrics \
  --out results/m4_thread_scan.json 2>&1 | tee results/m4_thread_scan.log
```
Read the printed table (decode tokens/s, joules per token, watts). Choose **N = the thread count with the best decode speed**; if two counts are
within about 3% of each other, take the smaller (less energy, less contention). Note the performance-core count (4) and whether
adding efficiency cores helps or hurts: that is a result worth reporting. Use this N everywhere below.

## 9. Generate and run the queue (about 18 hours)

**Size it from measurements, not hopes.** The Graviton rehearsal measured one evaluation at **88 s (stock) / 113 s (KleidiAI)** on Graviton3 (plan §23.2). Measuring all
256 genomes of both models in both configurations would need about 44 h at Graviton speed, so the default design is **reduced**: the full table in the
**`kai`** configuration (what ships) and **`stock`** only for the first 12 genomes (the reference, uniform baselines and the sensitivity-greedy chain), i.e.
`--prefix-config stock:12`. Projected hours for both models at a Mac that is 1x / 1.5x / 2x / 3x faster than Graviton3: 25.8 / 17.2 / 12.9 / 8.6 h.

1. **Find the Mac's speed.** Compare the selftest's decode speed and the thread scan's Q8_0 / Q4_0 numbers with Graviton3's (Phi-3 stock, 8 threads: Q8_0 36 tok/s,
   Q4_0 54 tok/s decode; prefill 93 / 109 tok/s; `results/rehearsal_thread_scan.json`). Expect the M4 to be faster for prefill (the KL run) and much less so for
   decode (memory bandwidth about 120 GB/s vs 155 GB/s measured on Graviton3). If the whole speedup is under 1.5x, do not plan both models in full.
2. **Divide the queue's hours** between the models: give each the hours the projection says it needs if both fit; otherwise cut Phi-3 first (it is the proxy
   for the story, the 8B model is the scale test) with `--share-phi3-hours`. Any prefix of the measurement order is a random sample, so partial tables remain analysable (GA studies need 90% coverage).
   Rerun the projection with your numbers: `python scripts/m4_timing_report.py --table results/rehearsal_table_phi3-mini.jsonl --queue-hours 18`.
3. **Generate and start it:**
```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/work/llama.cpp
# queue budget = 24 h - hours elapsed since allocating - 2.5 h for wrap-up; the shares must fit inside it
python scripts/run_queue.py --init-m4 queue_m4.json --budget-hours 18 --threads N --meter powermetrics \
    --share-8b-hours 11 --share-phi3-hours 5 --configs stock kai --prefix-config stock:12 \
    --sync-cmd aws s3 sync results s3://BUCKET/run1/results
python -m json.tool queue_m4.json | head -70            # read it: selftest first and required; tables flexible; --selftest and --prefix-config wired in
nohup caffeinate -dims python -u scripts/run_queue.py queue_m4.json > results/m4_queue.log 2>&1 &
tail -f results/m4_queue.log
```
(`--share-*` above are examples for a Mac about 1.5-2x faster than Graviton3: replace them with your sizing.)
What it does: re-runs the selftest as the required first job, then **table-llama3.1-8b** and **table-phi3-mini** (Q8_0/Q4_0 alphabet plus the F16 reference), each
stopping **before** a genome it cannot finish inside its share, then the two analyses (which always run, even past the deadline), and syncs `results/` to S3
after every job. Each table row is written as soon as it is measured; `--selftest` makes the table jobs honour the gate (a failed gate aborts, an unusable
configuration is dropped). Per-job logs: `results/queue_logs/<job>.log`. If you pause for a long time (a fix, a failed selftest), resume with
`--restart-clock HOURS` so the idle time does not eat the budget.

**While it runs** (look, do not touch): `tail -f results/queue_logs/table-llama3.1-8b.log`; progress `wc -l results/m4_table_*.jsonl`
(two rows per genome); machine state `pmset -g therm`; any `"error"` rows (`grep -c error results/m4_table_*.jsonl`). Do not run
other CPU work, do not open large applications, do not let the Mac sleep (`caffeinate` is in the command).
**Stop cleanly** after the current job: `touch queue_m4.state.STOP`. **Resume** after any interruption: run the same `nohup` line again; finished jobs are
skipped and the table driver continues from where it stopped (the deadline stays what it was).

## 10. Wrap-up (last 2.5 hours of the 24)

No S3 was set up on the first M4 day, so the desktop pulls everything with rsync; terminating the instance erases its disk
(`DeleteOnTermination`), so **nothing is final until the files are on the desktop and checked**.

**On the Mac** (the queue should have ended by itself; check with `tail -5 results/m4_queue.log` and `pgrep -fl "run_queue|m4_measure"`):
```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=.
touch queue_m4.state.STOP                       # only if it is still running: stops cleanly after the current job
python scripts/m4_analyze.py --table results/m4_table_llama3.1-8b.jsonl --model llama3.1-8b --out results/m4_analysis_llama3.1-8b.json | tail -40
python scripts/m4_analyze.py --table results/m4_table_phi3-mini.jsonl --model phi3-mini --out results/m4_analysis_phi3-mini.json | tail -40   # if not already done by the queue
cp pm_fixture.plist results/ ; wc -l results/m4_table_*.jsonl ; grep -c '"error"' results/m4_table_*.jsonl
```
**On the desktop** (copy only the Mac's files into a **separate folder**; a plain `rsync results/` into `results/` would overwrite tracked
evidence, and a source path without the trailing slash creates a nested `results/results/`):
```bash
rsync -avz -e "ssh -i ~/.ssh/KEY.pem" --include='m4_*' --include='queue_m4*' --include='pm_fixture.plist' --include='queue_logs/' --include='queue_logs/**' --exclude='*' \
  ec2-user@PUBLIC_IP:~/Evol_inference/results/ ~/work/Evol_inference/results_m4/
rsync -avz -e "ssh -i ~/.ssh/KEY.pem" ec2-user@PUBLIC_IP:~/Evol_inference/queue_m4.json ec2-user@PUBLIC_IP:~/Evol_inference/queue_m4.state.json ~/work/Evol_inference/results_m4/
wc -l ~/work/Evol_inference/results_m4/m4_table_*.jsonl            # must equal the Mac's counts above
```
(Run the same rsync every 2-3 hours during the day as insurance; it is cheap and repeatable.) Only after the counts match:
```bash
aws ec2 terminate-instances --region $R --instance-ids i-XXXXXXXX
```
The file `pm_fixture.plist` goes into `tests/fixtures/` (see section 12), not into `results/`.

## 11. Release the host (stops the billing)

Billing continues until the host is released. Release is refused until **24 hours after allocation** and until the instance is terminated.
```bash
aws ec2 describe-hosts --region $R --host-ids $HOST --query 'Hosts[0].[State,Instances]'
aws ec2 release-hosts --region $R --host-ids $HOST       # if "Unsuccessful": read the reason; retry after the 24 h mark or when scrubbing ends
```
Set a calendar reminder for allocation + 24 h. Check the console afterwards: host state `released`, no volumes, no running instance. The scrubbing
that follows termination takes up to 4.5 hours on Apple silicon; if release is refused for that reason, retry periodically.

## 12. What to do with the results (on the desktop)

```bash
cd ~/work/Evol_inference
PYTHONPATH=. .venv/bin/python scripts/m4_analyze.py --table results/m4_table_llama3.1-8b.jsonl --model llama3.1-8b     # front, GA vs greedy, kai vs stock
```
Add `!results/m4_*` to `.gitignore` (the Arm-port results are tracked), commit `pm_fixture.plist` and the Mac results, write the results into
the plan as §24, replace the "simulated" speed and energy in the deck's search slide with the measured front, and fix any runbook step that
did not survive contact with the Mac.

## 13. If something goes wrong

| Symptom | Likely cause / action |
|---|---|
| `allocate-hosts` fails with a quota error | the quota (section 1.1) is 0: request it; this can take days |
| "Insufficient capacity" on allocate | try another zone; capacity for Mac hosts is limited |
| instance stays `initializing` | macOS instances need 10-15 min; wait, then check the console screenshot |
| SSH refused / permission denied | wrong user (`ec2-user`), key, security group, or missing `-A` for a private repo |
| `sudo -n powermetrics` fails | needs passwordless sudo for that binary: `echo 'ec2-user ALL=(root) NOPASSWD: /usr/bin/powermetrics' | sudo tee /etc/sudoers.d/powermetrics` |
| powermetrics prints no `cpu_power` | update `POWER_KEYS_MW` in `energy_meter.py` to the real key (section 4) |
| `cmake` fails on an unknown option or on KleidiAI | record the error; try a build without `-DGGML_NATIVE=ON`; the stock build alone still lets the day produce a baseline |
| `cpu-only` check fails | Metal active: rebuild with `-DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF` |
| selftest `kai` unusable (SME garbage) | see section 7: continue with `stock` and `kai-nosme`, keep the evidence |
| memory pressure / swapping on the 8B | 8.5 GB model + page cache fits in 24 GiB; close everything else; check `memory_pressure` |
| speeds far below expectation or erratic | thermal throttling or Low Power Mode: `pmset -g therm`, `pmset -g`; Spotlight: `sudo mdutil -s /` |
| queue shows `skipped_budget` | the earlier jobs overran: the deadline is fixed; give the remaining jobs smaller shares and rerun with a new queue file |
| S3 sync fails | the queue continues (best effort); sync by hand at the end |
| the host is still billing the next day | it was not released: section 11 |
