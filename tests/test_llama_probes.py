"""llama.cpp probes with captured outputs and fake runners (tier T0)."""

import subprocess
from pathlib import Path

import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.llama_probes import (
    LlamaBenchProbe, LlamaPerplexityProbe, ProbeError, parse_llama_bench_json, parse_perplexity,
)

FIX = Path(__file__).parent / "fixtures"
GENOME = [Precision.INT8] * N_SUPER_BLOCKS


def cp(stdout="", stderr="", rc=0):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


def test_parse_perplexity_uses_final_estimate():
    assert parse_perplexity((FIX / "llama_perplexity_sample.txt").read_text()) == pytest.approx(4.3708)


def test_parse_perplexity_raises_without_final_line():
    with pytest.raises(ProbeError):
        parse_perplexity("[1]4.37,\n")


def test_perplexity_probe_command_and_parse():
    seen = []

    def runner(cmd):
        seen.append(cmd)
        return cp(stderr=(FIX / "llama_perplexity_sample.txt").read_text())

    probe = LlamaPerplexityProbe("llama-perplexity", "wiki.txt", lambda g: Path("/tmp/g.gguf"),
                                 threads=4, runner=runner)
    assert probe.perplexity(GENOME) == pytest.approx(4.3708)
    cmd = seen[0]
    assert cmd[cmd.index("-m") + 1] == "/tmp/g.gguf"
    assert cmd[cmd.index("-c") + 1] == "2048" and cmd[cmd.index("--chunks") + 1] == "1"
    assert cmd[cmd.index("-ngl") + 1] == "0" and cmd[cmd.index("-t") + 1] == "4"


def test_perplexity_probe_surfaces_failures():
    probe = LlamaPerplexityProbe("x", "f", lambda g: Path("g"), runner=lambda c: cp(stderr="boom", rc=1))
    with pytest.raises(ProbeError, match="boom"):
        probe.perplexity(GENOME)


def test_parse_bench_json_rows():
    rows = parse_llama_bench_json((FIX / "llama_bench_sample.json").read_text())
    assert rows[0].n_gen == 128 and rows[0].avg_ts == 40.0 and rows[0].samples_ts == (39.8, 40.0, 40.2)


def test_parse_bench_json_tolerates_log_noise_and_rejects_garbage():
    noisy = "load: warming up\n" + (FIX / "llama_bench_sample.json").read_text()
    assert parse_llama_bench_json(noisy)[0].n_gen == 128
    with pytest.raises(ProbeError):
        parse_llama_bench_json("not json")
    with pytest.raises(ProbeError, match="missing key"):
        parse_llama_bench_json('[{"n_prompt": 0}]')


class FakeMeasurement:
    def __init__(self, energy_j, duration_s=10.0, backend="powermetrics"):
        self.energy_j, self.duration_s, self.backend = energy_j, duration_s, backend

    @property
    def avg_power_w(self):
        return self.energy_j / self.duration_s

    def net_energy_j(self, idle_w):
        return max(0.0, self.energy_j - idle_w * self.duration_s)


def make_runner(decode_energy_j, backend="powermetrics", log=None):
    pp, tg = (FIX / "llama_bench_pp_sample.json").read_text(), (FIX / "llama_bench_sample.json").read_text()

    def run(cmd):
        if log is not None:
            log.append(cmd)
        is_pp = cmd[cmd.index("-p") + 1] != "0"
        e = decode_energy_j.pop(0) if not is_pp else 1.0
        return cp(stdout=pp if is_pp else tg), FakeMeasurement(e, backend=backend)

    return run


def test_bench_probe_total_energy_per_token():
    log = []
    probe = LlamaBenchProbe("llama-bench", lambda g: Path("g.gguf"), meter=None, idle_w=2.0, threads=4,
                            n_gen=128, repeats=3, runner=make_runner([120.0], log=log))
    m = probe.measure(GENOME)
    assert m.decode_tps == pytest.approx(40.0) and m.prefill_tps == pytest.approx(210.0)
    # net = 120 - 2*10 = 100 J over 128*3 tokens
    assert m.joules_per_token == pytest.approx(100.0 / (128 * 3))
    assert not m.synthetic
    assert m.decode_tps_spread == pytest.approx(0.2 / 40.0, rel=0.5)
    assert any(c[c.index("-n") + 1] == "0" for c in log) and any(c[c.index("-p") + 1] == "0" for c in log)
    assert all(c[c.index("-ngl") + 1] == "0" for c in log)  # CPU only


def test_bench_probe_differential_cancels_constant_overhead():
    # decode run (3 reps) = 30 J overhead + 3*40 J; low run (1 rep) = 30 + 40 -> diff 80 J over 2*128 tokens
    probe = LlamaBenchProbe("b", lambda g: Path("g"), None, idle_w=0.0, threads=4, n_gen=128, repeats=3,
                            energy_method="differential", runner=make_runner([150.0, 70.0]))
    m = probe.measure(GENOME)
    assert m.joules_per_token == pytest.approx(80.0 / (128 * 2))


def test_bench_probe_differential_rejects_non_positive_difference():
    probe = LlamaBenchProbe("b", lambda g: Path("g"), None, idle_w=0.0, threads=4, repeats=3,
                            energy_method="differential", runner=make_runner([50.0, 70.0]))
    with pytest.raises(ProbeError, match="non-positive"):
        probe.measure(GENOME)


def test_bench_probe_marks_mock_meter_results_synthetic():
    probe = LlamaBenchProbe("b", lambda g: Path("g"), None, idle_w=0.0, threads=4,
                            runner=make_runner([10.0], backend="mock"))
    assert probe.measure(GENOME).synthetic


def test_bench_probe_validates_method_and_surfaces_failure():
    with pytest.raises(ValueError):
        LlamaBenchProbe("b", lambda g: Path("g"), None, 0.0, 4, energy_method="bogus")
    probe = LlamaBenchProbe("b", lambda g: Path("g"), None, 0.0, 4,
                            runner=lambda c: (cp(stderr="oom", rc=1), FakeMeasurement(1.0)))
    with pytest.raises(ProbeError, match="oom"):
        probe.measure(GENOME)


# ---- real captured output (llama.cpp ec7630a, CPU build, Phi-3-mini) -------------------------

def test_parse_real_llama_cpp_perplexity_output():
    assert parse_perplexity((FIX / "llama_perplexity_real_ec7630a.txt").read_text()) == pytest.approx(5.2785)


def test_parse_real_llama_cpp_bench_json():
    rows = parse_llama_bench_json((FIX / "llama_bench_real_ec7630a.json").read_text())
    assert rows[0].n_prompt == 0 and rows[0].n_gen == 4
    assert rows[0].avg_ts > 0 and len(rows[0].samples_ts) == 2
