"""Prepare a fresh macOS (Apple silicon) host for the measurement day: toolchain, Python environment, two llama.cpp
builds (with and without KleidiAI), optional RAM disk, and the energy-meter prerequisites.

DRY RUN BY DEFAULT: prints the exact plan. `--execute` runs it (macOS only, stops at the first failure and writes
results/m4_bootstrap.json). The steps were written without access to a Mac and have NOT been run on one: rehearse the
plan on the host before the paid block starts, and expect to correct it (Doc/implementation_plan_mac-m4.md §23).

Build choices that matter for the comparison: Metal off (CPU only), Accelerate/BLAS off (so the stock build is the
llama.cpp NEON/I8MM path, not Apple's AMX/SME-backed BLAS), OpenMP off (Apple clang has none; ggml's own thread pool is
used). An optional third build (`--with-accelerate-build`) keeps Accelerate on, as another baseline.

    python scripts/bootstrap_mac.py                       # print the plan
    python scripts/bootstrap_mac.py --execute --ramdisk-gb 8
"""

from __future__ import annotations  # the macOS system Python is 3.9: `str | None` annotations must not be evaluated

import argparse
import json
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LLAMA_COMMIT = "ec7630a640789c393694fb194f1bbbf0369fc62d"  # pinned: every result so far was measured with this commit
PY_PKGS = ["numpy", "gguf", "datasets", "huggingface_hub", "transformers", "sentencepiece", "safetensors", "pytest", "torch"]
TARGETS = ["llama-perplexity", "llama-bench", "llama-cli", "llama-quantize"]


@dataclass
class Step:
    name: str
    cmd: str                 # shell command line
    why: str
    sudo: bool = False
    optional: bool = False


def ramdisk_sectors(gb: float) -> int:
    return int(gb * 2**30 / 512)


def cmake_flags(kleidiai: bool, accelerate: bool = False) -> list[str]:
    return ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=OFF", f"-DGGML_ACCELERATE={'ON' if accelerate else 'OFF'}",
            "-DGGML_BLAS=OFF", "-DGGML_OPENMP=OFF", f"-DGGML_CPU_KLEIDIAI={'ON' if kleidiai else 'OFF'}", "-DGGML_NATIVE=ON",
            "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF"]


def plan(llama_dir: str = "~/work/llama.cpp", ramdisk_gb: float = 0, with_accelerate: bool = False, repo: str = ".",
         s3_models: str | None = None, jobs: int = 10) -> list[Step]:
    ld = shlex.quote(llama_dir.replace("~", "$HOME")) if "~" not in llama_dir else llama_dir.replace("~", "$HOME")
    steps = [
        Step("record-machine", "sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize hw.ncpu hw.perflevel0.logicalcpu hw.perflevel1.logicalcpu; uname -m",
             "provenance: macOS version, chip, memory, P/E core counts (arm64 expected)"),
        Step("command-line-tools", "xcode-select -p && cc --version", "compiler: if this fails, install Xcode command line tools (interactive) first"),
        Step("homebrew", "command -v brew || NONINTERACTIVE=1 /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"",
             "package manager"),
        Step("brew-packages", "brew install cmake git python@3.12 awscli", "build tools, Python 3.12, AWS CLI for result sync"),
        Step("python-env", f"cd {repo} && $(brew --prefix python@3.12)/bin/python3.12 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install {' '.join(PY_PKGS)}",
             "project environment (CPU PyTorch is the default macOS arm64 wheel)"),
        Step("llama.cpp-clone", f"test -d {ld} || git clone https://github.com/ggml-org/llama.cpp {ld}; cd {ld} && git fetch --all -q && git checkout {LLAMA_COMMIT}",
             "pinned to the commit used for every earlier result"),
        Step("build-kai", f"cd {ld} && cmake -B build-kai {' '.join(cmake_flags(True))} && cmake --build build-kai -j {jobs} --target {' '.join(TARGETS)}",
             "KleidiAI on (SME2 kernels on M4), Metal/Accelerate/BLAS off"),
        Step("build-stock", f"cd {ld} && cmake -B build-stock {' '.join(cmake_flags(False))} && cmake --build build-stock -j {jobs} --target {' '.join(TARGETS[:3])}",
             "the fair speed baseline: llama.cpp's own Arm path, no KleidiAI"),
    ]
    if with_accelerate:
        steps.append(Step("build-stock-accelerate", f"cd {ld} && cmake -B build-stock-accelerate {' '.join(cmake_flags(False, True))} && cmake --build build-stock-accelerate -j {jobs} --target {' '.join(TARGETS[:3])}",
                          "optional extra baseline: Apple Accelerate BLAS enabled", optional=True))
    if ramdisk_gb:
        steps.append(Step("ramdisk", f"diskutil erasevolume HFS+ m4ram $(hdiutil attach -nomount ram://{ramdisk_sectors(ramdisk_gb)})",
                          f"{ramdisk_gb:g} GB RAM disk at /Volumes/m4ram for the assembled GGUF (24 GiB total memory: keep it below the model size)", optional=True))
    if s3_models:
        steps.append(Step("stage-models", f"cd {repo} && mkdir -p models/gguf && aws s3 sync {shlex.quote(s3_models)} models/gguf", "pre-staged GGUFs (F16/Q8_0/Q4_0 per model) from S3"))
    steps += [
        Step("powermetrics-sudo", "sudo -n powermetrics --samplers cpu_power -i 200 -n 2 -f plist > /dev/null",
             "the energy meter needs passwordless sudo for exactly this binary (sudoers: <user> ALL=(root) NOPASSWD: /usr/bin/powermetrics)", sudo=True),
        Step("low-power-mode-off", "sudo pmset -a lowpowermode 0", "Low Power Mode distorts speed and energy", sudo=True),
        Step("powermetrics-fixture", f"cd {repo} && . .venv/bin/activate && PYTHONPATH=. python energy_meter.py record pm_fixture.plist --seconds 10",
             "records real plist samples; prints the real key names if the parser's guesses are wrong; commit the file", sudo=True),
        Step("energy-selftest", f"cd {repo} && . .venv/bin/activate && PYTHONPATH=. ENERGY_BACKEND=powermetrics python energy_meter.py selftest",
             "idle vs busy power with the real backend"),
    ]
    return steps


def brew_env() -> dict:
    """PATH with Homebrew's Apple-silicon prefix first: right after the installer runs, /opt/homebrew/bin is not yet on PATH
    for later steps (it is added to the user's shell profile, which these non-interactive steps do not read)."""
    import os

    return dict(os.environ, PATH="/opt/homebrew/bin:/opt/homebrew/sbin:" + os.environ.get("PATH", ""))


def execute(steps: list[Step], report: Path) -> bool:
    done = []
    ok = True
    for s in steps:
        print(f"\n== {s.name}: {s.why}\n$ {s.cmd}", flush=True)
        t0 = time.time()
        rc = subprocess.run(["bash", "-lc", s.cmd], env=brew_env()).returncode
        done.append({"step": s.name, "rc": rc, "seconds": round(time.time() - t0, 1)})
        if rc != 0 and not s.optional:
            ok = False
            print(f"step {s.name} failed (rc {rc}); stopping", file=sys.stderr)
            break
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"ok": ok, "steps": done, "plan": [asdict(s) for s in steps]}, indent=2))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="run the plan (macOS only)")
    ap.add_argument("--force", action="store_true", help="allow --execute on a non-macOS host (rehearsal; will likely fail)")
    ap.add_argument("--llama-dir", default="~/work/llama.cpp")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ramdisk-gb", type=float, default=0)
    ap.add_argument("--with-accelerate-build", action="store_true")
    ap.add_argument("--s3-models", default=None, help="s3://bucket/prefix with the GGUFs")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--report", default="results/m4_bootstrap.json")
    a = ap.parse_args()
    steps = plan(a.llama_dir, a.ramdisk_gb, a.with_accelerate_build, a.repo, a.s3_models, a.jobs)
    if not a.execute:
        for i, s in enumerate(steps, 1):
            print(f"{i:2d}. {s.name}{' [sudo]' if s.sudo else ''}{' [optional]' if s.optional else ''}\n    {s.why}\n    $ {s.cmd}")
        print("\n(dry run: nothing executed; pass --execute on the Mac)")
        return
    if platform.system() != "Darwin" and not a.force:
        raise SystemExit("refusing to execute the macOS bootstrap on a non-macOS host (use --force to rehearse)")
    raise SystemExit(0 if execute(steps, Path(a.report)) else 1)


if __name__ == "__main__":
    main()
