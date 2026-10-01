"""energy_meter.py backends (tier T0): parser, baseline maths, RAPL wraparound, mock,
replay, the powermetrics streaming loop (against a fake `powermetrics` process), and
backend selection. Real-key verification happens against the recorded fixture on the Mac."""

import plistlib
import subprocess
import sys
import textwrap

import pytest

import energy_meter as em


def plist(**processor_and_top):
    top = {"elapsed_ns": processor_and_top.pop("elapsed_ns", 100_000_000)}
    top["processor"] = processor_and_top
    return plistlib.dumps(top)


# ---- parser -----------------------------------------------------------------------------

def test_parse_cpu_power_milliwatts_to_watts():
    assert em.parse_powermetrics_sample(plist(cpu_power=4500.0)) == pytest.approx(4.5)


def test_parse_falls_back_to_combined_power_then_energy():
    assert em.parse_powermetrics_sample(plist(combined_power=7000.0)) == pytest.approx(7.0)
    # 500 mJ over 0.1 s = 5 W
    assert em.parse_powermetrics_sample(plist(cpu_energy=500, elapsed_ns=100_000_000)) == pytest.approx(5.0)


def test_parse_prefers_cpu_power_over_combined():
    assert em.parse_powermetrics_sample(plist(cpu_power=1000.0, combined_power=9000.0)) == pytest.approx(1.0)


@pytest.mark.parametrize("doc", [b"", b"   \n", b"not a plist", plistlib.dumps({"processor": {}}),
                                 plistlib.dumps({"foo": 1})])
def test_parse_returns_none_for_unusable_samples(doc):
    assert em.parse_powermetrics_sample(doc) is None


# ---- Measurement ------------------------------------------------------------------------

def test_measurement_power_and_net_energy():
    m = em.Measurement("x", duration_s=10.0, energy_j=50.0)
    assert m.avg_power_w == pytest.approx(5.0)
    assert m.net_energy_j(2.0) == pytest.approx(30.0)
    assert m.net_energy_j(10.0) == 0.0  # clamped, never negative


def test_measurement_zero_duration():
    assert em.Measurement("x", 0.0, 1.0).avg_power_w == 0.0


def test_from_samples_energy_is_mean_power_times_duration():
    m = em._from_samples("x", 4.0, [2.0, 4.0])
    assert m.energy_j == pytest.approx(12.0) and m.avg_power_w == pytest.approx(3.0)
    assert em._from_samples("x", 4.0, []).energy_j == 0.0


# ---- RAPL -------------------------------------------------------------------------------

@pytest.fixture
def rapl_dir(tmp_path):
    (tmp_path / "max_energy_range_uj").write_text("1000000")  # 1 J range, tiny so we can wrap
    (tmp_path / "energy_uj").write_text("100000")
    return tmp_path


def test_rapl_energy_delta(rapl_dir):
    m = em.RaplMeter(str(rapl_dir))
    m.start()
    (rapl_dir / "energy_uj").write_text("400000")
    assert m.stop().energy_j == pytest.approx(0.3)


def test_rapl_counter_wraparound(rapl_dir):
    (rapl_dir / "energy_uj").write_text("900000")
    m = em.RaplMeter(str(rapl_dir))
    m.start()
    (rapl_dir / "energy_uj").write_text("200000")  # wrapped: 100000 to the top + 200000
    assert m.stop().energy_j == pytest.approx(0.3)


def test_rapl_missing_domain_raises(tmp_path):
    with pytest.raises(OSError):
        em.RaplMeter(str(tmp_path / "nope"))


# ---- mock -------------------------------------------------------------------------------

def test_mock_meter_is_deterministic_for_a_seed_and_responds_to_load():
    def reading(seed):
        m = em.MockMeter(seed=seed)
        _, meas = (m.measure_cmd([sys.executable, "-c", "sum(i*i for i in range(3_000_000))"]))
        return meas

    busy = reading(0)
    idle = em.MockMeter(seed=0)
    idle.start()
    import time
    time.sleep(busy.duration_s)
    idle_m = idle.stop()
    assert busy.avg_power_w > idle_m.avg_power_w
    assert idle_m.avg_power_w == pytest.approx(2.0, rel=0.15)  # idle_w=2 +/- noise


def test_measure_cmd_returns_process_output_and_still_stops_on_failure():
    m = em.MockMeter()
    proc, meas = m.measure_cmd([sys.executable, "-c", "print('hi')"])
    assert proc.stdout.strip() == "hi" and meas.duration_s > 0
    with pytest.raises(FileNotFoundError):
        m.measure_cmd(["definitely-not-a-binary-xyz"])


# ---- replay -----------------------------------------------------------------------------

@pytest.fixture
def fixture_file(tmp_path):
    docs = [plist(cpu_power=w * 1000.0) for w in (2.0, 4.0, 6.0)]
    p = tmp_path / "pm.plist"
    p.write_bytes(b"\0".join(docs))
    return str(p)


def test_replay_parses_fixture_and_cycles(fixture_file, capsys):
    m = em.ReplayMeter(fixture_file, interval_s=0.01)
    assert m._watts == [2.0, 4.0, 6.0]
    m.start()
    import time
    time.sleep(0.05)
    meas = m.stop()
    assert meas.backend == "replay" and meas.samples_w[0] == 2.0 and len(meas.samples_w) >= 3
    assert em._replay(fixture_file) == 0
    assert "parsed 3 samples" in capsys.readouterr().out


def test_replay_rejects_fixture_with_no_matching_keys(tmp_path):
    p = tmp_path / "bad.plist"
    p.write_bytes(plistlib.dumps({"processor": {"unknown_key": 1}}))
    with pytest.raises(ValueError, match="POWER_KEYS_MW"):
        em.ReplayMeter(str(p))


# ---- powermetrics streaming against a fake process ----------------------------------------

FAKE_PM = textwrap.dedent('''
    import plistlib, sys, time
    n = 0
    while True:
        sys.stdout.buffer.write(plistlib.dumps({"elapsed_ns": 50_000_000, "processor": {"cpu_power": 3000.0 + n}}) + b"\\0")
        sys.stdout.buffer.flush(); n += 1; time.sleep(0.02)
''')


def test_powermetrics_meter_streams_nul_separated_samples(tmp_path):
    script = tmp_path / "fake_pm.py"
    script.write_text(FAKE_PM)
    meter = em.PowermetricsMeter(interval_ms=50, sudo=False)
    meter.cmd = [sys.executable, str(script)]
    meter.start()
    import time
    time.sleep(0.4)
    m = meter.stop()
    assert m.backend == "powermetrics" and len(m.samples_w) >= 3
    assert 3.0 <= m.avg_power_w < 4.0
    assert m.energy_j == pytest.approx(m.avg_power_w * m.duration_s)


def test_powermetrics_meter_reports_stderr_when_no_samples(tmp_path):
    script = tmp_path / "bad_pm.py"
    script.write_text("import sys; sys.stderr.write('powermetrics must be invoked as the superuser'); sys.exit(1)")
    meter = em.PowermetricsMeter(interval_ms=20, sudo=False)
    meter.cmd = [sys.executable, str(script)]
    meter.start()
    with pytest.raises(RuntimeError, match="superuser"):
        meter.stop()


def test_powermetrics_cmd_uses_sudo_n_and_plist():
    cmd = em._powermetrics_cmd(100, sudo=True)
    assert cmd[:2] == ["sudo", "-n"] and "plist" in cmd and "cpu_power" in cmd
    assert em._powermetrics_cmd(100, sudo=False)[0] == "powermetrics"


# ---- backend selection ------------------------------------------------------------------

def test_get_meter_honours_env_and_explicit_backend(monkeypatch, fixture_file):
    monkeypatch.setenv("ENERGY_BACKEND", "mock")
    assert em.get_meter().name == "mock"
    monkeypatch.setenv("ENERGY_BACKEND", "replay")
    monkeypatch.setenv("ENERGY_FIXTURE", fixture_file)
    assert em.get_meter().name == "replay"
    assert em.get_meter("mock").name == "mock"
    with pytest.raises(KeyError):
        em.get_meter("bogus")


def test_autodetect_paths(monkeypatch):
    monkeypatch.setattr(em.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(em.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(em.shutil, "which", lambda x: "/usr/bin/powermetrics")
    assert em._autodetect() == "powermetrics"
    monkeypatch.setattr(em.platform, "system", lambda: "Linux")
    monkeypatch.setattr(em.os, "access", lambda p, m: True)
    assert em._autodetect() == "rapl"
    monkeypatch.setattr(em.os, "access", lambda p, m: False)
    assert em._autodetect() == "mock"


# ---- real RAPL, when readable --------------------------------------------------------------

def _rapl_readable():
    import os
    return os.access("/sys/class/powercap/intel-rapl:0/energy_uj", os.R_OK)


@pytest.mark.skipif(not _rapl_readable(), reason="RAPL energy_uj not readable on this host")
def test_real_rapl_busy_exceeds_idle():
    m = em.RaplMeter()
    base = m.idle_baseline(1.0)
    _, busy = m.measure_cmd([sys.executable, "-c", "sum(i*i for i in range(20_000_000))"])
    assert busy.avg_power_w > base
