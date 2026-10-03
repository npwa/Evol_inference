import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("m4_timing_report", Path("scripts/m4_timing_report.py"))
tr = importlib.util.module_from_spec(spec)
sys.modules["m4_timing_report"] = tr
spec.loader.exec_module(tr)


def row(genome, config, wall, bench, model="phi3-mini"):
    return {"genome": genome, "config": config, "model": model, "wall_s": wall, "bench_wall_s": bench}


ROWS = [row("88888888", "stock", 80, 45), row("44444444", "stock", 70, 38), row("84444444", "stock", 90, 40),
        row("88888888", "kai", 60, 35), row("44444444", "kai", 50, 30), row("84444444", "kai", 55, 31),
        row("FFFFFFFF", "stock", 400, 200), {"genome": "x", "config": "stock", "model": "phi3-mini", "error": "boom"}]


def test_summarize_separates_f16_genomes_and_ignores_errors():
    s = tr.summarize(ROWS)
    assert s["rows"] == 7 and set(s["configs"]) == {"stock", "kai"}
    st, kai = s["configs"]["stock"], s["configs"]["kai"]
    assert st["n_quantized"] == 3 and st["median_wall_s"] == 80 and st["median_bench_s"] == 40 and st["median_kld_s"] == 35   # wall - bench: median of 35, 32, 50
    assert st["n_with_f16"] == 1 and st["median_wall_s_with_f16"] == 400 and st["f16_genomes"] == ["FFFFFFFF"]
    assert kai["median_wall_s"] == 55 and "median_wall_s_with_f16" not in kai


def test_projection_scales_by_model_and_machine_speed():
    p = tr.project(tr.summarize(ROWS), hours=23, overhead_h=1.5)
    assert p["per_genome_all_configs_s"] == 135                     # 80 (stock) + 55 (kai)
    phi, big = p["models"]["phi3-mini"], p["models"]["llama3.1-8b"]
    assert phi["x1"]["full_space_h"] == pytest.approx(256 * 135 / 3600)
    assert big["x1"]["full_space_h"] == pytest.approx(phi["x1"]["full_space_h"] * 2.1)
    assert phi["x2"]["full_space_h"] == pytest.approx(phi["x1"]["full_space_h"] / 2)
    assert phi["x1"]["genomes_in_budget"] == int(21.5 * 3600 / 135)
    assert big["x1"]["genomes_in_budget"] < phi["x1"]["genomes_in_budget"]
