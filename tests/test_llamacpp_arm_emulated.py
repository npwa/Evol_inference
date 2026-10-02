"""Tier T2: the cross-compiled aarch64 llama.cpp (run under QEMU user-mode) must agree numerically with
the x86 build on the small proxy model. Correctness only. Slow (emulation): ~3 min per llama-perplexity
run; excluded from the default run with `-m "not slow"`.

Needs: ~/work/llama.cpp with build-cpu and build-aarch64-kaiOFF (generic NEON, no KleidiAI), qemu-aarch64,
the smollm2-360m GGUFs and models/gguf/wikitext2_head.txt (see scripts/t2_arm_emulated_check.py).
Tolerances come from measured values in plan §20 (F16 implementation difference ~1e-6 nats; Q4_0 within
~2% of x86)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from evol_inference.llama_probes import parse_kld, parse_perplexity

pytestmark = [pytest.mark.arm_emulated, pytest.mark.slow]

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp")).expanduser()
G = Path(os.environ.get("GGUF_DIR", "models/gguf"))
X86 = LLAMA / "build-cpu/bin/llama-perplexity"
ARM = LLAMA / "build-aarch64-kaiOFF/bin/llama-perplexity"
F16, Q4, TEXT = G / "smollm2-360m-f16.gguf", G / "smollm2-360m-q4_0-pure.gguf", G / "wikitext2_head.txt"

if not (X86.exists() and ARM.exists() and F16.exists() and Q4.exists() and TEXT.exists() and shutil.which("qemu-aarch64")):
    pytest.skip("needs x86 + aarch64 llama.cpp builds, qemu-aarch64 and the smollm2-360m GGUFs", allow_module_level=True)


def run(exe: Path, gguf: Path, *extra: str, cpu: str | None = None) -> str:
    env = dict(os.environ) | ({"QEMU_CPU": cpu} if cpu else {})
    p = subprocess.run([str(exe), "-m", str(gguf), "-f", str(TEXT), "-c", "256", "--chunks", "1", "-ngl", "0",
                        "-t", "2", *extra], capture_output=True, text=True, env=env)
    assert p.returncode == 0, (p.stdout + p.stderr)[-500:]
    return p.stdout + p.stderr


def test_aarch64_f16_matches_x86_logits(tmp_path):
    base = tmp_path / "x86_f16.kld"
    run(X86, F16, "--kl-divergence-base", str(base))
    out = run(ARM, F16, "--kl-divergence-base", str(base), "--kl-divergence", cpu="max")
    assert parse_kld(out).mean_kld < 1e-5   # measured ~1e-6: the F16 path is numerically the same


def test_aarch64_q4_0_accuracy_within_5_percent_of_x86(tmp_path):
    base = tmp_path / "x86_f16.kld"
    run(X86, F16, "--kl-divergence-base", str(base))
    x86 = parse_kld(run(X86, Q4, "--kl-divergence-base", str(base), "--kl-divergence")).mean_kld
    arm = parse_kld(run(ARM, Q4, "--kl-divergence-base", str(base), "--kl-divergence", cpu="neoverse-v1")).mean_kld
    assert arm == pytest.approx(x86, rel=0.05)   # measured 2.3% on this model (different dot-product implementations)
