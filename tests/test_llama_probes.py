"""llama.cpp probes with captured outputs and fake runners (tier T0)."""

import subprocess
from pathlib import Path

import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.llama_probes import (
    KldResult, LlamaBenchProbe, LlamaKldProbe, LlamaPerplexityProbe, ProbeError, parse_kld,
    parse_llama_bench_json, parse_perplexity,
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


# ---- KL divergence probe -------------------------------------------------------------------------

def test_parse_real_kld_output():
    r = parse_kld((FIX / "llama_kld_real_ec7630a.txt").read_text())
    assert r == KldResult(0.00165, 8.7e-05, 1.003812, 0.002337, 1.286, 98.436)


def test_parse_kld_requires_the_kld_line():
    with pytest.raises(ProbeError):
        parse_kld("Mean PPL(Q)/PPL(base) : 1.0 +- 0.1")


def kld_text(kld, ppl=None):
    return (f"Mean PPL(Q)/PPL(base)         :   1.003812 ±   0.002337\nMean    KLD:   {kld:.6f} ±   0.000087\n"
            "RMS Δp    :  1.286 ± 0.098 %\nSame top p: 98.436 ± 0.388 %\n")


def make_kld_probe(tmp_path, kld_by_name):
    log = []

    def runner(cmd):
        log.append(cmd)
        if "--kl-divergence" in cmd:
            return cp(stderr=kld_text(kld_by_name[Path(cmd[cmd.index("-m") + 1]).stem]))
        if "--kl-divergence-base" in cmd:
            (tmp_path / "base.kld").write_bytes(b"x")  # what --kl-divergence-base does
        return cp(stderr="Final estimate: PPL = 4.5000 +/- 0.3")

    name = lambda g: "".join(p.value[0] + p.value[-1] for p in g)  # noqa: E731
    probe = LlamaKldProbe("llama-perplexity", "wiki.txt", lambda g: Path("/tmp/" + name(g) + ".gguf"),
                          tmp_path / "base.kld", "/tmp/base_f16.gguf", runner=runner)
    return probe, log


def test_kld_probe_maps_kld_to_perplexity_equivalent(tmp_path):
    import math

    int8 = [Precision.INT8] * N_SUPER_BLOCKS
    probe, log = make_kld_probe(tmp_path, {"i8i8i8i8i8i8i8i8": 0.01})
    assert probe.perplexity(int8) == pytest.approx(4.5 * math.exp(0.01))
    assert (probe.perplexity(int8) - probe.ppl0) / probe.ppl0 == pytest.approx(math.expm1(0.01))
    # the first call saved the base model's logits; the measuring call compared against them
    assert log[0][log[0].index("-m") + 1] == "/tmp/base_f16.gguf" and "--kl-divergence-base" in log[0]
    assert "--kl-divergence" not in log[0]
    assert "--kl-divergence" in log[1] and log[1][log[1].index("--kl-divergence-base") + 1] == str(tmp_path / "base.kld")
    assert probe.history[tuple(p.value for p in int8)].mean_kld == 0.01


def test_kld_probe_reference_genome_is_measured_like_any_other(tmp_path):
    """The assembled all-reference genome is not the base model (its fixed tensors are quantized),
    so it gets a real, small, positive KLD instead of being assumed to be 0."""
    ref = [Precision.FP16] * N_SUPER_BLOCKS
    probe, log = make_kld_probe(tmp_path, {"f6f6f6f6f6f6f6f6": 0.0008})
    assert probe.perplexity(ref) > probe.ppl0


def test_kld_probe_reuses_existing_base_logits_file(tmp_path):
    (tmp_path / "base.kld").write_bytes(b"x")
    probe, log = make_kld_probe(tmp_path, {"i4i4i4i4i4i4i4i4": 0.05})
    probe.perplexity([Precision.INT4] * N_SUPER_BLOCKS)
    assert not any("--kl-divergence-base" in c and "--kl-divergence" not in c for c in log)  # base not rewritten
    assert log[0][log[0].index("-m") + 1] == "/tmp/base_f16.gguf"  # but ppl0 is still obtained


def test_parse_kld_handles_nan_stderr_for_identical_models():
    """A candidate identical to the base gives KLD 0 and undefined (-nan) standard errors."""
    text = ("Mean PPL(Q)/PPL(base)         :   1.000000 ±       -nan\n"
            "Mean    KLD:   0.000000 ±       -nan\nRMS Δp    :  0.000 ± -nan %\nSame top p: 100.000 ± 0.000 %\n")
    r = parse_kld(text)
    assert r.mean_kld == 0.0 and r.same_top_p == 100.0 and r.ppl_ratio == 1.0
    import math
    assert math.isnan(r.kld_stderr) and math.isnan(r.ppl_ratio_stderr)


# ---- kernel-path attribution ---------------------------------------------------------------------

SYSINFO_ARM = ("0.44.794.120 I system_info: n_threads = 4 (n_threads_batch = 4) / 16 | CPU : NEON = 1 | "
               "ARM_FMA = 1 | MATMUL_INT8 = 1 | LLAMAFILE = 1 | KLEIDIAI = 1 | REPACK = 1 | \n")


def test_parse_system_info_features():
    from evol_inference.llama_probes import parse_system_info

    assert parse_system_info(SYSINFO_ARM) == {"NEON": 1, "ARM_FMA": 1, "MATMUL_INT8": 1, "LLAMAFILE": 1,
                                              "KLEIDIAI": 1, "REPACK": 1}
    assert parse_system_info("nothing here") == {}
    real_x86 = (FIX / "llama_perplexity_real_ec7630a.txt").read_text()
    assert parse_system_info(real_x86)["AVX512"] == 1 and "NEON" not in parse_system_info(real_x86)


def test_parse_kleidiai_selection():
    from evol_inference.llama_probes import parse_kleidiai_selection

    log = ("0.1 I kleidiai: primary q4 kernel feature I8MM\n0.1 I kleidiai: primary q8 kernel feature DOTPROD\n"
           "0.1 I kleidiai: no compatible f32 kernels found for CPU features mask 3\n")
    assert parse_kleidiai_selection(log) == {"q4": "I8MM", "q8": "DOTPROD", "f32": None}
    assert parse_kleidiai_selection("no kleidiai lines at all") == {}


# ---- run configurations: extra args and environment -------------------------------------------------

def test_kld_probe_applies_extra_args_and_env(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output, text, env):
        seen["cmd"], seen["env"] = list(cmd), env
        return cp(stderr="Final estimate: PPL = 4.5 +/- 0.3")

    monkeypatch.setattr(subprocess, "run", fake_run)
    probe = LlamaKldProbe("llama-perplexity", "wiki.txt", lambda g: Path("g.gguf"), tmp_path / "b.kld", "base.gguf",
                          extra_args=["-nr"], env={"GGML_KLEIDIAI_SME": "0"})
    probe.ensure_base()
    assert "-nr" in seen["cmd"] and seen["env"]["GGML_KLEIDIAI_SME"] == "0" and "PATH" in seen["env"]


def test_bench_probe_applies_extra_args_and_env_through_the_meter():
    calls = []

    class Meter:
        def measure_cmd(self, cmd, **kw):
            calls.append((cmd, kw))
            tg = (FIX / "llama_bench_sample.json").read_text()
            pp = (FIX / "llama_bench_pp_sample.json").read_text()
            return cp(stdout=pp if cmd[cmd.index("-p") + 1] != "0" else tg), FakeMeasurement(100.0)

    probe = LlamaBenchProbe("llama-bench", lambda g: Path("g.gguf"), Meter(), idle_w=0.0, threads=4,
                            extra_args=["--repack", "0"], env={"GGML_KLEIDIAI_SME": "0"})
    probe.measure(GENOME)
    assert all(c[0][-2:] == ["--repack", "0"] for c in calls)
    assert all(kw["env"]["GGML_KLEIDIAI_SME"] == "0" for _, kw in calls)
