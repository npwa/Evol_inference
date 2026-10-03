from evol_inference.selftest_gates import (
    Check, gate_backend_is_cpu, gate_energy, gate_f16_floor, gate_machine_state, gate_platform, gate_q4, gate_q8,
    gate_resources, gate_speed, overall_ok, usable_configs,
)


def test_platform_gate():
    assert gate_platform("Darwin", "arm64", False).ok
    assert not gate_platform("Linux", "x86_64", False).ok
    assert gate_platform("Linux", "x86_64", True).ok          # rehearsal allowed
    assert not gate_platform("Darwin", "x86_64", False).ok


def test_q8_gate_accepts_the_expected_kleidiai_loss_and_rejects_garbage():
    stock = 0.000755
    assert gate_q8("kai", 0.0283, stock)[0].ok                 # Graviton3 Phi-3: 37x, expected
    assert gate_q8("kai", 0.0327, 0.00389)[0].ok               # proxy: 8x
    assert gate_q8("kai", 0.0008, stock)[0].ok                 # no loss at all is fine
    assert not gate_q8("kai", 23.1, 0.00389)[0].ok             # SME under QEMU: ~5,900x
    assert "WRONG KERNEL" in gate_q8("kai", 23.1, 0.00389)[0].detail
    assert "EXPECTED" in gate_q8("kai", 0.0283, stock)[0].detail


def test_q4_gate_is_a_warning():
    ok = gate_q4("kai", 0.085873, 0.085873)
    bad = gate_q4("kai", 0.2, 0.085873)
    assert ok.ok and not bad.ok and bad.severity == "warn" and not bad.failed_fatally


def test_energy_gate_requires_response_to_load_only_for_real_meters():
    assert gate_energy("powermetrics", 2.0, 9.0)[0].ok
    flat = gate_energy("powermetrics", 2.0, 2.1)[0]
    assert not flat.ok and flat.failed_fatally
    mock = gate_energy("mock", 2.0, 2.0)[0]
    assert mock.ok or mock.severity == "warn"                   # a synthetic backend never aborts a rehearsal
    assert not gate_energy("powermetrics", 2.0, 9.0, fixture_present=False)[1].ok     # missing fixture: warning
    assert len(gate_energy("rapl", 2.0, 9.0)) == 1


def test_cpu_only_gate_catches_metal():
    assert gate_backend_is_cpu("CPU").ok and gate_backend_is_cpu("BLAS,CPU").ok
    assert not gate_backend_is_cpu("Metal,CPU").ok and not gate_backend_is_cpu("MTL").ok and not gate_backend_is_cpu("CUDA").ok


def test_machine_state_gate_low_power_is_fatal_throttle_is_warning():
    lpm, therm = gate_machine_state(False, False, True, "")[0], gate_machine_state(False, True, False, "x")[1]
    assert lpm.failed_fatally and not therm.ok and not therm.failed_fatally


def test_resources_and_floor():
    assert all(c.ok for c in gate_resources(200, 24, 60, 16))
    assert not gate_resources(30, 24, 60, 16)[0].ok and not gate_resources(200, 8, 60, 16)[1].ok
    assert gate_f16_floor(1e-6, "kai").ok and not gate_f16_floor(1e-2, "kai").ok
    assert gate_speed("kai", 40, 38).ok and not gate_speed("kai", 5, 38).ok


def test_usable_configs_and_overall_decision():
    configs = ["stock", "kai", "kai-nosme"]
    checks = gate_q8("kai", 23.1, 0.004) + gate_q8("kai-nosme", 0.03, 0.004) + gate_q8("stock", 0.004, 0.004)
    assert usable_configs(checks, configs) == ["stock", "kai-nosme"]       # kai (SME path) dropped, run continues without it
    assert overall_ok(checks, configs)
    assert not overall_ok(gate_q8("stock", 30.0, 0.00001), ["stock", "kai"])  # stock itself garbage: abort
    assert not overall_ok([Check("disk", False, "fatal", "full")], configs)   # a global fatal check aborts
    assert overall_ok([], ["stock"])


def test_cpu_only_gate_names_the_configuration():
    assert gate_backend_is_cpu("Metal,CPU", "kai").name == "cpu-only[kai]"
    assert usable_configs([gate_backend_is_cpu("Metal,CPU", "kai")], ["stock", "kai"]) == ["stock"]


def test_disk_gate_names_the_location_and_hints_at_tmpfs_when_it_fails():
    ok = gate_resources(30, 24, 1.0, 16, where="/home/ubuntu/work")[0]
    bad = gate_resources(7.7, 24, 16.0, 16, where="/tmp/m4_selftest")[0]
    assert ok.ok and "/home/ubuntu/work" in ok.detail
    assert not bad.ok and bad.failed_fatally and "/tmp/m4_selftest" in bad.detail and "tmpfs" in bad.detail and "--work-dir" in bad.detail
