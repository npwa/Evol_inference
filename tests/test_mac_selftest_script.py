"""scripts/mac_selftest.py end to end: the failure path (cheap) and a rehearsal pass with real binaries (marker llamacpp)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp")).expanduser()
GG = Path(os.environ.get("GGUF_DIR", "models/gguf"))
HAVE = (LLAMA / "build-cpu/bin/llama-bench").exists() and (GG / "smollm2-360m-f16.gguf").exists() and (GG / "wikitext2_head.txt").exists()
ENV = dict(os.environ, PYTHONPATH=".", LLAMA_CPP_DIR=str(LLAMA))


def selftest(tmp_path, *args):
    out = tmp_path / "st.json"
    p = subprocess.run([sys.executable, "scripts/mac_selftest.py", "--model", "smollm2-360m", "--allow-non-mac", "--meter", "mock",
                        "--idle-seconds", "0.5", "--out", str(out), "--work-dir", str(tmp_path / "w"), *args],
                       capture_output=True, text=True, env=ENV)
    return p, (json.loads(out.read_text()) if out.exists() else None)


@pytest.mark.skipif(not (GG / "smollm2-360m-f16.gguf").exists(), reason="needs the smollm2-360m GGUFs")
def test_selftest_fails_closed_when_a_build_is_missing(tmp_path):
    p, rep = selftest(tmp_path, "--configs", "stock", "kai", "--build-kai", "no-such-build", "--build-stock", "no-such-build")
    assert p.returncode == 1 and "SELFTEST FAILED" in p.stdout
    assert rep["ok"] is False and rep["usable_configs"] == []
    assert any(c["name"] == "files-present" and not c["ok"] for c in rep["checks"])


@pytest.mark.llamacpp
@pytest.mark.skipif(not HAVE, reason="needs llama.cpp build-cpu, the smollm2-360m GGUFs and wikitext2_head.txt")
def test_selftest_rehearsal_passes_and_records_kernel_info(tmp_path):
    p, rep = selftest(tmp_path, "--configs", "stock", "kai-nr", "--build-kai", "build-cpu", "--build-stock", "build-cpu",
                      "--text", str(GG / "wikitext2_head.txt"), "--ctx", "256", "--threads", "4")
    assert p.returncode == 0, p.stdout[-800:]
    assert rep["ok"] and set(rep["usable_configs"]) == {"stock", "kai-nr"}
    assert rep["energy"]["backend"] == "mock" and rep["configs"]["stock"]["kld"]["q4"] > 0.1
    assert rep["platform"]["machine"] and "machine_raw" in rep
