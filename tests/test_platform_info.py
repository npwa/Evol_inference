from pathlib import Path

import pytest

from evol_inference import platform_info as pi

FIX = Path(__file__).parent / "fixtures"


def test_parse_sysctl_m4():
    info = pi.parse_sysctl((FIX / "sysctl_m4_sample.txt").read_text())
    assert info["model"] == "Apple M4"
    assert info["memory_gib"] == 24.0
    assert info["core_topology"] == {"performance_cores": 4, "efficiency_cores": 6}
    f = info["arm_features"]
    assert f["i8mm"] and f["dotprod"] and f["sme"] and f["sme2"] and f["neon"]
    assert not f["sve"]  # M4 has no SVE


def test_parse_cpuinfo_graviton3_has_sve_i8mm_but_no_sme():
    info = pi.parse_proc_cpuinfo((FIX / "cpuinfo_graviton3_sample.txt").read_text())
    f = info["arm_features"]
    assert f["sve"] and f["i8mm"] and f["bf16"] and f["dotprod"] and f["neon"]
    assert not f["sme"] and not f["sve2"]


def test_parse_cpuinfo_x86_has_model_and_no_arm_features():
    info = pi.parse_proc_cpuinfo((FIX / "cpuinfo_x86_sample.txt").read_text())
    assert "i7-11700K" in info["model"]
    assert not any(info["arm_features"].values())


def test_collect_has_required_fields_on_this_host():
    info = pi.collect(extra={"kleidiai": False})
    assert info["system"] and info["machine"] and info["kleidiai"] is False
    assert isinstance(info["is_arm"], bool)
    pi.require_platform({pi.PLATFORM_KEY: info})


@pytest.mark.parametrize("bad", [{}, {"platform": None}, {"platform": {}}, {"platform": "x"}])
def test_require_platform_rejects_results_without_provenance(bad):
    with pytest.raises(ValueError, match="platform"):
        pi.require_platform(bad)
