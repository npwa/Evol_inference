"""Parsers and the throttle wait loop (formats from memory; see evol_inference/mac_env.py)."""

from evol_inference.mac_env import MachineState, machine_state, parse_lowpowermode, parse_pmset_therm, wait_until_cool

QUIET = """Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
"""
THROTTLED = """CPU_Scheduler_Limit	= 100
CPU_Available_CPUs	= 10
CPU_Speed_Limit		= 78
"""
PMSET = """System-wide power settings:
Currently in use:
 standby              1
 lowpowermode         0
 sleep                1
"""


def test_quiet_machine_is_not_throttled():
    t = parse_pmset_therm(QUIET)
    assert not t["throttled"] and t["warnings"] == []


def test_speed_limit_below_100_is_throttling():
    t = parse_pmset_therm(THROTTLED)
    assert t["throttled"] and t["limits"]["CPU_Speed_Limit"] == 78


def test_recorded_warning_level_is_throttling():
    assert parse_pmset_therm("Note: Thermal warning level 1 recorded\n")["throttled"]


def test_lowpowermode_parse():
    assert parse_lowpowermode(PMSET) is False
    assert parse_lowpowermode(PMSET.replace("lowpowermode         0", "lowpowermode         1")) is True
    assert parse_lowpowermode("nothing") is None


def test_machine_state_darwin_flags_problems_and_keeps_raw_output():
    outputs = {("pmset", "-g", "therm"): THROTTLED, ("pmset", "-g"): PMSET.replace("lowpowermode         0", "lowpowermode         1")}
    st = machine_state(run=lambda cmd: outputs[tuple(cmd)], system="Darwin")
    assert not st.ok and st.throttled and st.low_power_mode is True
    assert "Low Power Mode" in st.detail and "pmset -g therm" in st.raw


def test_machine_state_elsewhere_is_na_and_ok():
    st = machine_state(system="Linux")
    assert st.ok and st.detail.startswith("n/a")


def test_wait_until_cool_returns_when_state_clears_and_respects_deadline():
    seq = iter([MachineState("Darwin", throttled=True), MachineState("Darwin", throttled=True), MachineState("Darwin", throttled=False)])
    t = [0.0]
    st = wait_until_cool(60, 5, state_fn=lambda: next(seq), sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert not st.throttled and t[0] == 10  # polled twice

    t2 = [0.0]
    st2 = wait_until_cool(20, 5, state_fn=lambda: MachineState("Darwin", throttled=True), sleep=lambda s: t2.__setitem__(0, t2[0] + s), clock=lambda: t2[0])
    assert st2.throttled and t2[0] == 20  # gave up at the deadline, did not hang
