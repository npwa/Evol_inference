"""kernels/arm (qdot): quantized GEMV kernels. The portable reference is tested natively on any host
(tier T0); the NEON SDOT / SMMLA kernels are tested BIT-EXACT against it in aarch64 emulation (tier T2:
aarch64 cross-compiler + qemu-user needed, skipped otherwise)."""

import shutil
import subprocess
from pathlib import Path

import pytest

KERNELS = Path(__file__).resolve().parents[1] / "kernels" / "arm"
TOOLCHAIN = KERNELS / "toolchain-aarch64.cmake"


def _build(tmp_path: Path, *cmake_args: str) -> Path:
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(KERNELS), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", *cmake_args],
                   check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(build), "-j4", "--target", "qdot_test"], check=True, capture_output=True)
    return build / "qdot_test"


@pytest.mark.skipif(shutil.which("cmake") is None or shutil.which("g++") is None, reason="needs cmake and a C++ compiler")
def test_reference_kernels_native(tmp_path):
    out = subprocess.run([str(_build(tmp_path))], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-800:]
    assert "PASS" in out.stdout and "0 failures" in out.stdout


_HAVE_ARM = shutil.which("aarch64-linux-gnu-g++") is not None and shutil.which("qemu-aarch64") is not None


@pytest.mark.arm_emulated
@pytest.mark.skipif(not _HAVE_ARM, reason="needs aarch64-linux-gnu-g++ and qemu-aarch64 (tier T2)")
@pytest.mark.parametrize("cpu,expect_i8mm", [("neoverse-n1", False), ("neoverse-v1", True), ("max", True)])
def test_neon_kernels_bit_exact_under_qemu(tmp_path, cpu, expect_i8mm):
    exe = _build(tmp_path, f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN}")
    out = subprocess.run(["qemu-aarch64", "-cpu", cpu, str(exe)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-800:]
    assert "0 failures" in out.stdout
    assert "sdot=1" in out.stdout                       # the SDOT path really ran
    assert f"i8mm={int(expect_i8mm)}" in out.stdout     # SMMLA only where the emulated CPU has I8MM
