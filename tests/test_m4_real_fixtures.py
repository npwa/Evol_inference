"""The real Apple M4 outputs recorded on the first mac-m4.metal day (macOS 26.7, Mac16,10) run through the
parsers that were written from memory. tests/fixtures/m4_real/ holds the raw captures."""

from pathlib import Path

import energy_meter as em
from evol_inference.mac_env import parse_lowpowermode, parse_pmset_therm

FIX = Path(__file__).parent / "fixtures" / "m4_real"


def test_real_powermetrics_plist_parses_cpu_power_in_milliwatts():
    raw = (FIX / "powermetrics.plist").read_bytes()
    watts = [w for w in map(em.parse_powermetrics_sample, raw.split(b"\0")) if w is not None]
    assert watts == [0.0237641, 0.149888, 0.0900962]      # cpu_power 23.7641 / 149.888 / 90.0962 mW


def test_real_powermetrics_plist_has_the_assumed_keys():
    raw = (FIX / "powermetrics.plist").read_bytes()
    for key in (b"<key>cpu_power</key>", b"<key>cpu_energy</key>", b"<key>combined_power</key>"):
        assert raw.count(key) == 3                          # three samples
    assert em.POWER_KEYS_MW[0] == "cpu_power"


def test_replay_meter_runs_on_the_real_fixture(capsys):
    assert em._replay(str(FIX / "powermetrics.plist")) == 0
    assert "parsed 3 samples" in capsys.readouterr().out


def test_real_pmset_therm_is_the_quiet_case():
    t = parse_pmset_therm((FIX / "pmset_therm.txt").read_text())
    assert t["throttled"] is False and t["limits"] == {} and t["warnings"] == []


def test_real_pmset_reports_low_power_mode_off():
    assert parse_lowpowermode((FIX / "pmset.txt").read_text()) is False


def test_real_sysctl_records_the_m4_core_layout_and_sme2():
    lines = (FIX / "sysctl.txt").read_text().splitlines()
    assert lines[:5] == ["Apple M4", "25769803776", "10", "4", "6"]   # brand, memsize, ncpu, performance, efficiency cores
    assert "hw.optional.arm.FEAT_SME2: 1" in lines
    assert "hw.optional.arm.FEAT_I8MM: 1" in lines
