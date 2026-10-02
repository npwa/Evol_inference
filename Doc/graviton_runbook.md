# Runbook: tier T3 on an AWS Graviton instance

Purpose: the first measurements on **real Arm silicon**, at a cost of a few dollars, before the 24-hour
Apple M4 block (tier T4). Everything here is run on the instance except where marked **[desktop]**.
Background and the findings this run follows up: `Doc/implementation_plan_mac-m4.md` §20.

**What this run answers**
1. Do the NEON SDOT / SMMLA kernels (`kernels/arm/qdot`) pass bit-exact on real hardware (and the fp16
   conversion match the hardware)? (T2 showed it under emulation only.)
2. KleidiAI **on vs off** for Phi-3-mini: accuracy (KL divergence) and speed (prefill and decode
   tokens/s at several thread counts). T2 found KleidiAI's Q8_0 path ~8x less accurate on a 360M proxy; is it
   the same on Phi-3?
3. Which KleidiAI kernels does the real CPU select (log with `-v`)?
4. Memory bandwidth (roofline) and where the decode kernel sits against it.
5. Whether hardware performance counters (`perf`) are available on this instance type.

Time about 3 hours, cost about 1-2 USD on-demand. **No power data**: Graviton exposes no energy counter
(power needs the Mac, T4).

---

## 1. Instance type

| Type | CPU | vCPU / RAM | Verdict |
|---|---|---|---|
| `t4g.small` | Graviton2, Neoverse-N1 (NEON + dotprod only; no SVE, no I8MM, no BF16) | 2 / 2 GiB, burstable | **No.** 2 GiB cannot load Phi-3 (F16 7.6 GB, Q8_0 4.1 GB), CPU-credit throttling makes timings meaningless, and it exercises only the DOTPROD path. |
| **`c7g.2xlarge`** | **Graviton3, Neoverse-V1: NEON, dotprod, I8MM, BF16, SVE (256-bit)** | 8 / 16 GiB | **Recommended.** Same feature set as the QEMU `neoverse-v1` model used in T2; ~0.29 USD/h on-demand (check the current price in your region). Fits Phi-3 F16 for the reference logits (7.6 GB) with room to spare. |
| `c8g.2xlarge` | Graviton4, Neoverse-V2: NEON, dotprod, I8MM, BF16, SVE2 (128-bit vectors) | 8 / 16 GiB | **Optional second run** (~0.32 USD/h): KleidiAI's SVE kernel needs 256-bit vectors, so on this CPU it should pick I8MM instead. Same steps, different instance. |
| `c7g.metal` | Graviton3 | 64 / 128 GiB | Only if `perf` counters are unavailable on the virtualized instance (see step 12); about 2.3 USD/h, run for minutes. |

Use **c7g.2xlarge**, Ubuntu, 64 GiB gp3 root volume (see the resource table below), **on-demand** (spot is fine for a re-runnable job
but an interruption loses the session). Any region that offers c7g; pick the one nearest to you.

### Resources needed (measured / computed for the Phi-3 run)

| Resource | Needed | Why |
|---|---|---|
| **vCPUs** | **8 recommended**, 4 workable | One Graviton vCPU is one physical core (no SMT). The thread scan uses 1, 4 and 8 threads, and the two llama.cpp builds compile with `-j8`. With 4 vCPUs everything still runs, but builds and the accuracy runs take about twice as long and the 8-thread point is lost. |
| **Memory** | **16 GiB recommended**, 12 GiB minimum | The heaviest step, Phi-3 F16 perplexity at 2048 tokens, peaks at **8.5 GB resident** (measured on the desktop: weights 7.6 GB + KV cache 0.8 GB + logits 0.3 GB). The rest of the memory serves as page cache so the 7.6 GB file stays cached across runs. 8 GiB (`c7g.xlarge`) is **not** enough for the F16 reference run. |
| **Disk** | **64 GiB** (about 32 GiB used) | Ubuntu + build tools about 8 GB; HF safetensors 7.2 GB (needed only for the conversion, deletable afterwards); GGUFs 13.9 GB (F16 7.6 + Q8_0 4.1 + Q4_0 2.2); Python environment with CPU PyTorch about 1.5 GB; llama.cpp source and both builds about 0.6 GB (each build is only 150-190 MB); KL-divergence base logits 65 MB; results are tiny. If you later also run Llama-3.1-8B on the same instance (+44 GB), use **100 GiB**. |
| **Disk throughput** | gp3 at **500 MB/s** (default is 125) | A cold read of the 7.6 GB F16 file takes about 60 s at 125 MB/s and about 15 s at 500 MB/s; the extra throughput costs cents for a few hours. Set it at launch (included in the CLI command below) or later under Volumes > Modify. |
| **Network** | outbound internet | Hugging Face download (7.2 GB), the llama.cpp clone, KleidiAI fetched by CMake, PyTorch wheels. |

The same-family alternative `m7g.xlarge` (Graviton3, 4 vCPU / 16 GiB) is the cheapest instance that satisfies the
memory requirement; `c7g.2xlarge` (8 vCPU / 16 GiB) is recommended for the thread scan and build speed.

## 2. Operating system image

* **Recommended: Ubuntu Server 24.04 LTS, 64-bit Arm.** It is what the development desktop runs, so the
  toolchain (gcc 13, Python 3.12, CMake 3.28) and every Python wheel used so far are known to work.
* **Ubuntu 26.04 LTS (64-bit Arm) should work for the C++ parts** (llama.cpp, the kernels), but I could not
  verify from here that the AMI exists in your account/region, nor that PyTorch publishes aarch64 wheels for
  its default Python version; the model-conversion step (step 8, option A) depends on that. If you want
  26.04, either use step 8 option B (copy the GGUFs from the desktop) or install Python 3.12 with `uv`.

Look up the current official AMI id (needs the AWS CLI configured on the desktop; read-only call):
```bash
# [desktop]  24.04 (recommended)
aws ssm get-parameter --region REGION \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text
# [desktop]  26.04 (if Canonical has published it for your region; an error means it has not)
aws ssm get-parameter --region REGION \
  --name /aws/service/canonical/ubuntu/server/26.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text
```
(Console alternative: EC2 > Launch instance > Quick Start > Ubuntu > choose the **64-bit (Arm)** architecture.)

## 3. Launch **[desktop]**

Console: EC2 > Launch instance. Name `evol-graviton`; the AMI above; instance type `c7g.2xlarge`; select (or
create) a key pair; network: allow **SSH (22) from your IP only**; storage **64 GiB gp3** (set throughput to 500 MB/s, see below); launch.

CLI equivalent:
```bash
MYIP=$(curl -s https://checkip.amazonaws.com)
aws ec2 create-security-group --region REGION --group-name evol-graviton-ssh --description "ssh from my ip"
aws ec2 authorize-security-group-ingress --region REGION --group-name evol-graviton-ssh \
  --protocol tcp --port 22 --cidr ${MYIP}/32
aws ec2 run-instances --region REGION --image-id AMI_ID --instance-type c7g.2xlarge \
  --key-name YOUR_KEY --security-groups evol-graviton-ssh \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=64,VolumeType=gp3,Iops=3000,Throughput=500,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=evol-graviton}]'
```
Note the public IP (`PUBIP`) and instance id (`i-...`). Connect:
```bash
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBIP          # public repository
ssh -A -i ~/.ssh/YOUR_KEY.pem ubuntu@PUBIP       # private repository: -A forwards your desktop's ssh-agent (see step 5)
```
(`-A` lets the instance use your GitHub key through the agent **without copying the private key onto it**;
make sure the key is loaded on the desktop first: `ssh-add -l` should list it, otherwise `ssh-add ~/.ssh/id_ed25519`.)

## 4. System packages (on the instance)

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-pip rsync tmux htop \
     "linux-tools-$(uname -r)" || sudo apt-get install -y linux-tools-aws
lscpu | grep -E "Model name|Architecture|^CPU\(s\)"        # expect aarch64, Neoverse-V1, 8 CPUs
grep -m1 Features /proc/cpuinfo                            # expect asimddp, i8mm, bf16, sve in the list
python3 --version; gcc --version | head -1; cmake --version | head -1
```
Work inside `tmux` (`tmux new -s t3`) so a dropped connection does not kill long steps.

## 5. Project code (on the instance)

The `mac-m4` branch is on the remote (`git@github.com:npwa/Evol_inference.git`), so clone it there. Which
form depends on whether the repository is public or private; try the first, and if it asks for a username or
returns "not found", the repository is private and you need the second.

```bash
# (a) public repository: HTTPS, no credentials
git clone --branch mac-m4 --single-branch https://github.com/npwa/Evol_inference.git ~/Evol_inference

# (b) private repository: SSH through the forwarded agent (connect with `ssh -A`, see step 3)
mkdir -p ~/.ssh && ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
ssh -T git@github.com                      # expect: "Hi npwa! You've successfully authenticated ..."
git clone --branch mac-m4 --single-branch git@github.com:npwa/Evol_inference.git ~/Evol_inference
```
Alternative for (b) if you prefer not to forward the agent: create a **read-only fine-grained personal access
token** for this one repository on GitHub (Contents: read), then
`git clone --branch mac-m4 --single-branch https://github.com/npwa/Evol_inference.git ~/Evol_inference` and
enter the token as the password. Delete the token afterwards. Never put a private key or token in a file that
is committed.

Record exactly what you are testing (the commit is stored with the results in step 10's log header):
```bash
cd ~/Evol_inference && git log -1 --format='%H %s' | tee ~/t3_commit.txt
```
Then the Python environment:
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy gguf datasets huggingface_hub transformers sentencepiece safetensors pytest
mkdir -p models/gguf results
export PYTHONPATH=.
```
(`bitsandbytes` and CUDA are not needed on Arm; the GPU tests skip themselves.)

## 6. Build llama.cpp (pinned to the commit used everywhere else)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout ec7630a640789c393694fb194f1bbbf0369fc62d
# with KleidiAI, tuned to this CPU (-mcpu=native picks up SVE / I8MM / BF16)
cmake -B build-kai -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-kai -j8 --target llama-perplexity llama-bench llama-cli llama-quantize
# optional belt-and-braces: a build without KleidiAI (the run-time switch --no-repack gave identical numbers in T2)
cmake -B build-nokai -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build-nokai -j8 --target llama-perplexity llama-bench llama-cli
```
Each build takes roughly 10-20 minutes. Skip `build-nokai` to save time and drop `--build-nokai` in step 10.

Check what the build detected:
```bash
~/llama.cpp/build-kai/bin/llama-bench --help >/dev/null && echo "binary runs"
```

## 7. Kernel library on real hardware (about 5 minutes)

```bash
cd ~/Evol_inference
cmake -S kernels/arm -B build-kernels -DCMAKE_BUILD_TYPE=Release
cmake --build build-kernels -j8
./build-kernels/hwcap_probe            # sdot, i8mm, bf16, sve = yes; sme, sme2 = no (on c7g)
./build-kernels/qdot_test | tee results/t3_qdot_test.txt
```
**Expected:** `aarch64: sdot=1 i8mm=1` and `PASS: 108 shape/mode cases, 0 failures`. Anything else is a
real finding (the NEON kernels or the fp16 conversion disagree with hardware): stop and record the output.

Memory bandwidth and kernel throughput (single thread; Phi-3's four weight-matrix shapes `M x K`):
```bash
./build-kernels/bw_probe 1 8 | tee results/t3_bw_probe.jsonl
for s in "9216 3072" "3072 3072" "16384 3072" "3072 8192"; do ./build-kernels/qdot_bench $s 30; done \
  | tee results/t3_qdot_bench.jsonl
```

## 8. Models: Phi-3-mini GGUFs

**Option A (recommended; no transfer from home): download and convert on the instance (about 15 minutes).**
```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=.
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("microsoft/Phi-3-mini-4k-instruct", local_dir="models/phi-3-mini-4k-instruct",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model", "*.py", "LICENSE*"])
EOF
python ~/llama.cpp/convert_hf_to_gguf.py models/phi-3-mini-4k-instruct --outtype f16 \
       --outfile models/gguf/phi3-mini-f16.gguf
Q=~/llama.cpp/build-kai/bin/llama-quantize
$Q --pure models/gguf/phi3-mini-f16.gguf models/gguf/phi3-mini-q8_0-pure.gguf Q8_0 8
$Q --pure models/gguf/phi3-mini-f16.gguf models/gguf/phi3-mini-q4_0-pure.gguf Q4_0 8
python -c "from evol_inference.eval_data import write_wikitext2_test; write_wikitext2_test('models/gguf/wikitext2_test.txt')"
```
**Option B (if conversion fails, e.g. on a newer Python): copy the finished files [desktop].**
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/YOUR_KEY.pem" \
  ~/work/Evol_inference/models/gguf/phi3-mini-{f16,q8_0-pure,q4_0-pure}.gguf \
  ~/work/Evol_inference/models/gguf/wikitext2_test.txt ubuntu@PUBIP:~/Evol_inference/models/gguf/
```
**Verify the files are the same as on the desktop** (quantization is deterministic, so option A should match):
```bash
cd ~/Evol_inference/models/gguf && sha256sum phi3-mini-f16.gguf phi3-mini-q8_0-pure.gguf phi3-mini-q4_0-pure.gguf
```
```
f53d06d3c9c7e8a5638d08ec21381f2a39076bb817c8110688a306da3c5fc5a2  phi3-mini-f16.gguf
49a323d0b284896256bcb741c5ca60fee85ac8268f9d878cf97bc6cc150340f4  phi3-mini-q8_0-pure.gguf
033f660b977fec6bb9ded43150d04d211f48ae2553afe03a9bf5c46f2ef8afc0  phi3-mini-q4_0-pure.gguf
```
A mismatch in the F16 file would be a conversion-environment difference (different library versions): note it
and continue; a mismatch in the Q8_0/Q4_0 files given a matching F16 would be a real finding.

## 9. Unit tests on the instance (about 1 minute)

```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=.
python -m pytest -q -m "not gpu and not llamacpp and not slow"
```
Expect everything to pass; the aarch64-emulation tests skip (no QEMU) and `test_reference_kernels_native`
now builds and runs the NEON kernels natively.

## 10. The main run: KleidiAI on / off, accuracy and speed (about 1-1.5 hours)

```bash
cd ~/Evol_inference && . .venv/bin/activate && export PYTHONPATH=. LLAMA_CPP_DIR=~/llama.cpp
(cat ~/t3_commit.txt; nohup python -u scripts/t3_native_arm_check.py --model phi3-mini \
      --build-kai build-kai --build-nokai build-nokai --threads 1 4 8) \
      > results/t3_main.log 2>&1 &
tail -f results/t3_main.log
```
(Drop `--build-nokai build-nokai` if you skipped that build.) The script, for each of `kai` (KleidiAI default),
`kai-nr` (same build, `--no-repack`) and `nokai`: measures KL divergence of Q8_0 and Q4_0 against the F16
logits at a 2048-token context, records the kernel selection from a `-v` run, and runs `llama-bench`
(prefill 512, decode 128, 3 repetitions) for F16/Q8_0/Q4_0 at 1, 4 and 8 threads. Output:
`results/t3_native_phi3-mini_<hostname>.json` (with a platform block). To shorten it, use
`--skip-bench` (accuracy only, ~25 minutes) or `--threads 8`.

**What to look for**
* `kai` vs `kai-nr` Q8_0 KLD: the factor between them is the cost of KleidiAI's per-row Q8_0 re-quantization on
  Phi-3 (it was ~8x on the 360M proxy). Q4_0 should be equal to within a few percent.
* `kleidiai` kernel selection: expected on c7g `q4: SVE`, `q8: I8MM` (what QEMU's `neoverse-v1` showed).
* `tg128` (decode) tokens/s for `kai` vs `kai-nr`, per precision and thread count; decode should approach
  bandwidth-bound scaling (compare with `results/t3_bw_probe.jsonl`).
* `kai-nr` and `nokai` agree (confirms the run-time switch).

## 11. Optional: second CPU generation

Repeat steps 1-10 on a `c8g.2xlarge` (Graviton4) and compare the kernel selection (expect `q4: I8MM`,
because the SVE kernel needs 256-bit vectors) and the speed.

## 12. Optional: hardware performance counters

```bash
sudo sysctl kernel.perf_event_paranoid=-1
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
  -- ~/llama.cpp/build-kai/bin/llama-bench -m ~/Evol_inference/models/gguf/phi3-mini-q4_0-pure.gguf \
     -p 0 -n 128 -r 2 -t 8 -ngl 0 2>&1 | tee ~/Evol_inference/results/t3_perf_stat.txt
```
If the events show `<not supported>` the virtualized instance does not expose the PMU; counters would need
`c7g.metal` (run just this step there, for a few minutes). Record which it was either way.

## 13. Bring results back and shut down

```bash
# [desktop]
rsync -avz -e "ssh -i ~/.ssh/YOUR_KEY.pem" ubuntu@PUBIP:~/Evol_inference/results/ ~/work/Evol_inference/results/
# [desktop]  terminate: stops billing and deletes the volume. Do this as soon as results are copied.
aws ec2 terminate-instances --region REGION --instance-ids i-XXXXXXXX
```
Check in the console that the instance shows `terminated` and that no volume or Elastic IP remains.
**Cost control:** an on-demand instance bills until terminated; consider an AWS budget alarm at ~10 USD.

## 14. If something goes wrong

| Symptom | Likely cause / action |
|---|---|
| `Illegal instruction` | binary built on a different CPU generation, or `-mcpu=native` mismatch: rebuild on the instance. |
| `qdot_test` FAIL | record the full output; this is a real result (kernel or fp16 conversion vs hardware). |
| `kleidiai` selection empty | KleidiAI only logs with `-v` and only once it sees Q4_0/Q8_0 weights; the script already does this. |
| `git clone` asks for a password / "repository not found" | the repository is private: use `ssh -A` (step 3) and form (b) in step 5, or a read-only token. |
| `Permission denied (publickey)` on `git clone` | the key is not loaded in your desktop agent (`ssh-add -l`), or you connected without `-A`. |
| conversion step fails | use option B (copy GGUFs). |
| `InsufficientInstanceCapacity` | try another availability zone or region, or `c7g.xlarge` (4 vCPU, 8 GiB: skip the F16 reference run, `--skip-accuracy`). |
| cmake cannot fetch KleidiAI | the instance needs outbound internet (default VPC has it). |

After the run, results go into the plan as §21, and the Mac selftest gates (plan §20.4) are tuned with the
Graviton numbers.
