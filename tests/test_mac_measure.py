"""Mac measurement tooling: ordering, persistence, measured-table probes, and the offline analyzer on a synthetic
256-genome table (tier T0); one integration test with real llama.cpp binaries and the proxy model (marker llamacpp)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.mac_measure import (
    append_row, done_keys, greedy_chain, load_rows, priority_order, standard_configs,
)
from evol_inference.measured_table import MeasuredTable
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.model_spec import get_spec
from evol_inference.probes import SimulatedArmProbe
from evol_inference.sensitivity import SensitivityTable
from evol_inference.tabulated import enumerate_genomes, genome_code, parse_code

SPEC8 = get_spec("llama3.1-8b")
BLOCK_COST = [1.234, 0.431, 0.468, 0.534, 0.471, 0.346, 0.358, 0.864]  # % per block at Q4_0 (the measured 8B profile)


def synthetic_rows(config="stock", speed_scale=1.0, kld_scale=1.0, synthetic=False, base_ppl=6.79):
    arm = SimulatedArmProbe.graviton3([452_984_832] * N_SUPER_BLOCKS, fixed_bytes=5.6e8, noise=0)
    rows = []
    for g in enumerate_genomes(SPEC8.alphabet):
        se = arm.measure(g)
        kld = kld_scale * sum(c / 100 for c, p in zip(BLOCK_COST, g) if p is Precision.INT4)
        rows.append({"genome": genome_code(g), "config": config, "model": SPEC8.key, "kld": kld, "kld_stderr": 1e-5, "same_top_p": 95.0,
                     "decode_tps": se.decode_tps * speed_scale, "prefill_tps": se.prefill_tps, "decode_spread": 0.01,
                     "joules_per_token": se.joules_per_token, "avg_power_w": 8.0, "synthetic": synthetic, "base_ppl": base_ppl})
    return rows


# ---- ordering -------------------------------------------------------------------------------------------

def test_priority_order_baselines_first_then_greedy_chain_then_everything_once():
    sens = SensitivityTable(6.79, "int8", {"int4": BLOCK_COST})
    order = priority_order(SPEC8, None, sens, seed=0)
    codes = [genome_code(g) for g in order]
    assert len(codes) == len(set(codes)) == 256
    assert codes[:3] == ["88888888", "44444444", "84444448"]  # uniform Q8, uniform Q4, heuristic
    chain = [genome_code(g) for g in greedy_chain(SPEC8, sens)]
    assert chain[1] == "44444484" or chain[1].count("8") == 1  # k=1: only the most sensitive block (block 0) at Q8_0
    assert set(chain) <= set(codes[: 3 + len(chain)])           # the chain follows immediately after the baselines


def test_greedy_chain_protects_most_sensitive_blocks_first():
    sens = SensitivityTable(6.79, "int8", {"int4": BLOCK_COST})
    chain = greedy_chain(SPEC8, sens)
    assert len(chain) == N_SUPER_BLOCKS + 1
    assert genome_code(chain[0]) == "44444444" and genome_code(chain[-1]) == "88888888"
    assert genome_code(chain[1]) == "84444444"   # block 0 (1.234) is the most sensitive
    assert genome_code(chain[2]) == "84444448"   # then block 7 (0.864)


def test_priority_order_is_seeded_and_a_prefix_is_a_random_sample():
    a, b, c = (priority_order(SPEC8, None, None, seed=s) for s in (0, 0, 1))
    assert [genome_code(g) for g in a] == [genome_code(g) for g in b]
    assert [genome_code(g) for g in a][5:30] != [genome_code(g) for g in c][5:30]


def test_priority_order_respects_alphabet_restriction_but_always_measures_the_reference_first():
    spec = get_spec("phi3-mini")
    order = priority_order(spec, [Precision.INT8, Precision.INT4], None)
    assert genome_code(order[0]) == "FFFFFFFF"                               # the F16 reference, outside the alphabet
    assert len(order) == 257 and all(set(g) <= {Precision.INT8, Precision.INT4} for g in order[1:])
    assert len(priority_order(spec, [Precision.INT8, Precision.INT4], None, include_reference=False)) == 256


def test_assembly_space_is_the_largest_source_among_the_precisions_in_play(tmp_path):
    from evol_inference.mac_measure import assembly_space_gb

    src = {}
    for p, size in ((Precision.FP16, 7_600_000), (Precision.INT8, 4_000_000), (Precision.INT4, 2_000_000)):
        f = tmp_path / f"{p.value}.gguf"
        f.write_bytes(b"\0" * size)
        src[p] = f
    assert assembly_space_gb(src, {Precision.INT8, Precision.INT4}) == pytest.approx(0.004)     # the F16 file is irrelevant here
    assert assembly_space_gb(src, {Precision.FP16, Precision.INT4}) == pytest.approx(0.0076)
    with pytest.raises(ValueError):
        assembly_space_gb(src, set())


def test_standard_configs():
    cfg = standard_configs("build-kai", "build-stock")
    assert set(cfg) == {"kai", "kai-nosme", "kai-nr", "stock"}
    assert dict(cfg["kai-nosme"].env) == {"GGML_KLEIDIAI_SME": "0"} and cfg["kai-nr"].ppl_args == ("-nr",)
    assert cfg["kai-nr"].bench_args == ("--repack", "0") and "stock" not in standard_configs("build-kai", None)


# ---- persistence ------------------------------------------------------------------------------------------

def test_rows_roundtrip_and_error_rows_are_not_done(tmp_path):
    p = tmp_path / "t.jsonl"
    append_row(p, {"genome": "88888888", "config": "kai", "kld": 0.1})
    append_row(p, {"genome": "44444444", "config": "kai", "error": "boom"})
    assert len(load_rows(p)) == 2 and done_keys(load_rows(p)) == {("88888888", "kai")}
    assert load_rows(tmp_path / "missing.jsonl") == []


# ---- measured table ------------------------------------------------------------------------------------------

def test_measured_table_median_of_repeats_and_probe_interface():
    rows = synthetic_rows()[:1] * 3
    rows = [dict(r, decode_tps=v) for r, v in zip(rows, (10.0, 30.0, 20.0))]
    t = MeasuredTable(rows, "stock")
    g = parse_code(rows[0]["genome"])
    assert t.measure(g).decode_tps == 20.0 and t.repeats(g) == 3 and len(t) == 1
    assert t.perplexity(g) == pytest.approx(6.79 * math.exp(rows[0]["kld"]))


def test_measured_table_rejects_unmeasured_genome_and_mixed_bases():
    t = MeasuredTable(synthetic_rows()[:2], "stock")
    with pytest.raises(KeyError, match="not measured"):
        t.kld(parse_code("44444444"))
    mixed = [dict(r) for r in synthetic_rows()[:2]]
    mixed[1]["base_ppl"] = 9.99
    with pytest.raises(ValueError, match="different base models"):
        MeasuredTable(mixed, "stock")
    with pytest.raises(ValueError, match="no usable rows"):
        MeasuredTable(synthetic_rows()[:2], "kai")


def test_synthetic_flag_propagates():
    assert MeasuredTable(synthetic_rows(synthetic=True)[:3], "stock").synthetic
    assert not MeasuredTable(synthetic_rows(synthetic=False)[:3], "stock").synthetic


def test_evaluator_on_a_measured_table_reproduces_the_reference_as_zero():
    t = MeasuredTable(synthetic_rows(), "stock")
    ev = MultiObjectiveEvaluator(t, t, SPEC8.reference_genome(), baseline_repeats=1)
    v = ev.evaluate(SPEC8.reference_genome())
    assert v.accuracy_penalty == pytest.approx(0) and v.speed_gain == pytest.approx(0) and v.energy_gain == pytest.approx(0)
    q4 = ev.evaluate([Precision.INT4] * N_SUPER_BLOCKS)
    assert q4.speed_gain > 0.3 and q4.accuracy_penalty > 0.04   # Q4_0 faster and less accurate than Q8_0


# ---- analyzer on a synthetic full table ----------------------------------------------------------------------

def test_analyzer_recovers_the_known_front_and_reports_config_ratios(tmp_path):
    table = tmp_path / "t.jsonl"
    for r in synthetic_rows("stock") + synthetic_rows("kai", speed_scale=1.5, kld_scale=2.0):
        append_row(table, r)
    sens = tmp_path / "sens.json"
    SensitivityTable(6.79, "int8", {"int4": BLOCK_COST}).to_json(sens)
    out = subprocess.run([sys.executable, "scripts/m4_analyze.py", "--table", str(table), "--model", "llama3.1-8b",
                          "--sensitivity", str(sens), "--studies", "3", "--out", str(tmp_path / "r.json")],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert out.returncode == 0, out.stderr[-600:]
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["baseline_config"] == "stock" and not rep["synthetic"]
    for c in ("stock", "kai"):
        e = rep["configs"][c]
        assert e["measured"] == 256 and e["front_size"] == 9               # the 9-point greedy chain
        assert e["greedy"]["hv_ratio"] == pytest.approx(1.0)   # the greedy chain IS the front
        assert e["ga_hv_ratio"]["min"] > 0.999            # the GA gets (almost) all of it, as in §19
    x = rep["cross"]["kai/stock"]
    assert x["decode_median"] == pytest.approx(1.5) and x["kld_ratio_median"] == pytest.approx(2.0, rel=0.05)


def test_analyzer_falls_back_to_a_measured_uniform_baseline_when_the_reference_is_missing(tmp_path):
    """Rehearsal bug: a Q8_0/Q4_0-only table for Phi-3 has no F16 reference row. The analyzer must warn, not crash."""
    table = tmp_path / "t.jsonl"
    for r in synthetic_rows("stock")[:60]:
        append_row(table, dict(r, model="phi3-mini"))
    out = subprocess.run([sys.executable, "scripts/m4_analyze.py", "--table", str(table), "--model", "phi3-mini"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert out.returncode == 0, out.stderr[-500:]
    assert "WARNING" in out.stdout and "FFFFFFFF was not measured" in out.stdout and "baseline is 88888888" in out.stdout


def test_analyzer_labels_mock_energy_results_synthetic(tmp_path):
    table = tmp_path / "t.jsonl"
    for r in synthetic_rows("stock", synthetic=True)[:40]:
        append_row(table, r)
    out = subprocess.run([sys.executable, "scripts/m4_analyze.py", "--table", str(table), "--model", "llama3.1-8b"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert out.returncode == 0 and "SYNTHETIC" in out.stdout and "skipped" in out.stdout   # partial table: studies skipped


# ---- integration with real binaries (proxy model, mock meter) ---------------------------------------------------

LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", "~/work/llama.cpp")).expanduser()
GG = Path(os.environ.get("GGUF_DIR", "models/gguf"))


@pytest.mark.llamacpp
@pytest.mark.skipif(not ((LLAMA / "build-cpu/bin/llama-bench").exists() and (GG / "smollm2-360m-f16.gguf").exists()
                         and (GG / "wikitext2_head.txt").exists()), reason="needs llama.cpp build-cpu and the smollm2-360m GGUFs")
def test_measurer_end_to_end_with_real_binaries(tmp_path):
    from energy_meter import MockMeter
    from evol_inference.dryrun_setup import find_sources
    from evol_inference.mac_measure import GenomeMeasurer, MeasureSettings, RunConfig

    spec = get_spec("smollm2-360m")
    cfgs = [RunConfig("stock", "build-cpu"), RunConfig("kai-nr", "build-cpu", ppl_args=("-nr",), bench_args=("--repack", "0"))]
    m = GenomeMeasurer(spec, find_sources(spec, GG), LLAMA, cfgs, GG / "wikitext2_head.txt", tmp_path, MockMeter(),
                       MeasureSettings(threads=4, ctx=256, n_prompt=128, n_gen=32, repeats=1, cooldown_wait_s=0), idle_s=0.5)
    row = m.measure(spec.reference_genome(), "stock")
    row2 = m.measure(spec.reference_genome(), "kai-nr")
    assert row["genome"] == "F" * 8 and 0 < row["kld"] < 0.01 and row["decode_tps"] > 0 and row["joules_per_token"] > 0
    assert row["synthetic"] is True and row["energy_backend"] == "mock" and row["base_ppl"] == pytest.approx(26.89, rel=0.01)
    assert row2["kld"] == pytest.approx(row["kld"], rel=0.05)   # same genome, KleidiAI-off flag: same accuracy on x86


def test_measure_table_honours_the_selftest_verdict(tmp_path):
    """A failed gate aborts; an unusable configuration is dropped, not measured (runs before any model is touched)."""
    def run(st):
        f = tmp_path / "st.json"
        f.write_text(json.dumps(st))
        return subprocess.run([sys.executable, "scripts/m4_measure_table.py", "--model", "smollm2-360m", "--selftest", str(f),
                               "--configs", "stock", "kai", "--limit", "0", "--out", str(tmp_path / "t.jsonl"), "--work-dir", str(tmp_path / "w")],
                              capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    failed = run({"ok": False, "usable_configs": []})
    assert failed.returncode != 0 and "did not pass" in (failed.stdout + failed.stderr)
    none = run({"ok": True, "usable_configs": ["other"]})
    assert none.returncode != 0 and "no usable configuration" in (none.stdout + none.stderr)


# ---- prefix configurations and budget-aware stopping (found by the first Graviton rehearsal) ------------------------

def test_prefix_limits_parse_and_select_configs_by_position():
    from evol_inference.mac_measure import configs_for, parse_prefix_limits

    lim = parse_prefix_limits(["stock:12", "kai-nr:3"])
    assert lim == {"stock": 12, "kai-nr": 3} and parse_prefix_limits(None) == {} and parse_prefix_limits([]) == {}
    labels = ["stock", "kai", "kai-nr"]
    assert configs_for(0, labels, lim) == ["stock", "kai", "kai-nr"]
    assert configs_for(3, labels, lim) == ["stock", "kai"]                   # past kai-nr's prefix
    assert configs_for(12, labels, lim) == ["kai"]                           # past stock's prefix: only the primary configuration
    for bad in ("stock", "stock:x", ":4"):
        with pytest.raises(ValueError):
            parse_prefix_limits([bad])


def test_should_stop_ends_before_a_genome_that_would_overrun_the_budget():
    from evol_inference.mac_measure import should_stop

    assert not should_stop(100, None, [50])                                   # no budget: never
    assert should_stop(1001, 1000, [])                                        # plain budget check without history
    assert not should_stop(100, 1000, [])
    assert not should_stop(800, 1000, [100, 100, 100])                        # 800 + 1.15 * 100 < 1000
    assert should_stop(900, 1000, [100, 100, 100])                            # 900 + 115 > 1000: the next genome would overrun
    assert should_stop(500, 1000, [100, 100, 100], next_has_f16=True) is False   # 500 + 1.15*300 = 845
    assert should_stop(700, 1000, [100, 100, 100], next_has_f16=True)         # an F16 genome costs ~3x: 700 + 345 > 1000
    assert not should_stop(900, 1000, [10, 10, 5000])                         # the median, not an outlier, is the estimate


def test_analyzer_alphabet_option_restricts_the_space_to_the_measured_levels(tmp_path):
    """First M4 day: the Phi-3 table was measured over {Q8_0, Q4_0} (256 genomes), but the analyzer counted the 3-level space (6561)
    and skipped the GA / greedy studies at 4% 'coverage'. --alphabet int8 int4 fixes the space."""
    table = tmp_path / "t.jsonl"
    for r in synthetic_rows("kai"):
        append_row(table, dict(r, model="phi3-mini"))
    cmd = [sys.executable, "scripts/m4_analyze.py", "--table", str(table), "--model", "phi3-mini", "--studies", "2", "--out", str(tmp_path / "a.json")]
    full = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert full.returncode == 0 and "256/6561 genomes (4%)" in full.stdout and "studies skipped" in full.stdout
    restricted = subprocess.run(cmd + ["--alphabet", "int8", "int4"], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert restricted.returncode == 0, restricted.stderr[-500:]
    assert "256/256 genomes (100%)" in restricted.stdout and "GA (400 evals" in restricted.stdout
    rep = json.loads((tmp_path / "a.json").read_text())
    assert rep["space"] == 256 and rep["alphabet"] == ["int8", "int4"]


def test_analyzer_skips_the_ga_study_when_a_nearly_complete_table_misses_genomes(tmp_path):
    """First M4 day: 249 of 256 genomes (97%) passed the 90% gate, but the GA draws random genomes and crashed on an unmeasured one."""
    table = tmp_path / "t.jsonl"
    for r in synthetic_rows("kai")[:-3]:
        append_row(table, r)
    out = subprocess.run([sys.executable, "scripts/m4_analyze.py", "--table", str(table), "--model", "llama3.1-8b", "--studies", "2"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="."))
    assert out.returncode == 0, out.stderr[-500:]
    assert "253/256 genomes" in out.stdout and "3 missing" in out.stdout and "GA (400 evals" not in out.stdout
