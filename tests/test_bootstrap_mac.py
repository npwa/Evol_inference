"""Bootstrap plan content and safety (the plan is generated and tested here; it has NOT been executed on a Mac)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("bootstrap_mac", Path("scripts/bootstrap_mac.py"))
bm = importlib.util.module_from_spec(spec)
sys.modules["bootstrap_mac"] = bm
spec.loader.exec_module(bm)


def test_builds_use_the_flags_the_comparison_needs():
    kai, stock = bm.cmake_flags(True), bm.cmake_flags(False)
    for f in ("-DGGML_METAL=OFF", "-DGGML_ACCELERATE=OFF", "-DGGML_BLAS=OFF", "-DGGML_OPENMP=OFF", "-DGGML_NATIVE=ON"):
        assert f in kai and f in stock
    assert "-DGGML_CPU_KLEIDIAI=ON" in kai and "-DGGML_CPU_KLEIDIAI=OFF" in stock
    assert "-DGGML_ACCELERATE=ON" in bm.cmake_flags(False, accelerate=True)


def test_plan_pins_the_llama_cpp_commit_and_builds_both_variants():
    steps = {s.name: s for s in bm.plan()}
    assert bm.LLAMA_COMMIT in steps["llama.cpp-clone"].cmd
    assert "build-kai" in steps["build-kai"].cmd and "build-stock" in steps["build-stock"].cmd
    assert "KLEIDIAI=ON" in steps["build-kai"].cmd and "KLEIDIAI=OFF" in steps["build-stock"].cmd
    assert "build-stock-accelerate" not in steps and "ramdisk" not in steps
    assert steps["powermetrics-sudo"].sudo and "powermetrics" in steps["powermetrics-sudo"].cmd


def test_optional_steps_appear_only_when_requested():
    steps = {s.name: s for s in bm.plan(ramdisk_gb=8, with_accelerate=True, s3_models="s3://b/models")}
    assert steps["ramdisk"].optional and "ram://16777216" in steps["ramdisk"].cmd     # 8 GiB / 512 B sectors
    assert "ACCELERATE=ON" in steps["build-stock-accelerate"].cmd and "aws s3 sync" in steps["stage-models"].cmd
    assert bm.ramdisk_sectors(1) == 2097152


def test_dry_run_is_the_default_and_executes_nothing():
    p = subprocess.run([sys.executable, "scripts/bootstrap_mac.py"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert p.returncode == 0 and "dry run" in p.stdout and "brew install" in p.stdout


def test_refuses_to_execute_on_a_non_mac_host():
    if sys.platform == "darwin":
        return
    p = subprocess.run([sys.executable, "scripts/bootstrap_mac.py", "--execute"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert p.returncode != 0 and "non-macOS" in (p.stdout + p.stderr)


def test_execute_stops_at_the_first_required_failure_and_continues_past_optional_ones(tmp_path):
    steps = [bm.Step("one", "true", "x"), bm.Step("opt", "false", "x", optional=True), bm.Step("bad", "exit 3", "x"), bm.Step("never", "true", "x")]
    ok = bm.execute(steps, tmp_path / "r.json")
    rep = json.loads((tmp_path / "r.json").read_text())
    assert not ok and [s["step"] for s in rep["steps"]] == ["one", "opt", "bad"] and rep["steps"][2]["rc"] == 3
    assert bm.execute([bm.Step("a", "true", "x")], tmp_path / "r2.json")
