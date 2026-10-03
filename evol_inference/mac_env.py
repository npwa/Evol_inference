"""Machine-state checks for the Mac (T4) measurement day: thermal throttling and Low Power Mode distort
speed and energy, so they are checked before each measurement and recorded with every result.

Parsing is separated from the (macOS-only) commands so it is unit-tested on any host. **The `pmset`
output formats below are from memory and unverified**: the selftest stores the raw text of every command
(`MachineState.raw`) so the first minute on a real Mac produces fixtures to correct these parsers against.
On a non-macOS host every check reports state "n/a" and passes, so the same driver runs on the desktop
and on Graviton for rehearsal."""

from __future__ import annotations

import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

Runner = Callable[[list[str]], str]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_pmset_therm(text: str) -> dict:
    """`pmset -g therm`. Keys of interest when throttling is active: CPU_Scheduler_Limit, CPU_Available_CPUs,
    CPU_Speed_Limit (percent). The quiet case prints 'Note: No thermal warning level has been recorded' etc."""
    limits = {k: int(v) for k, v in re.findall(r"(CPU_Speed_Limit|CPU_Scheduler_Limit|CPU_Available_CPUs)\s*=\s*(\d+)", text)}
    recorded = [ln for ln in text.splitlines()
                if re.search(r"warning level|power status", ln, re.I) and not re.search(r"\bNo\b", ln)]
    throttled = limits.get("CPU_Speed_Limit", 100) < 100 or limits.get("CPU_Scheduler_Limit", 100) < 100 or bool(recorded)
    return {"limits": limits, "warnings": recorded, "throttled": throttled}


def parse_lowpowermode(text: str) -> bool | None:
    """`pmset -g` lists ' lowpowermode  0|1'. None if the line is absent."""
    m = re.search(r"^\s*lowpowermode\s+(\d)", text, re.M)
    return None if m is None else m.group(1) == "1"


@dataclass
class MachineState:
    system: str
    ok: bool = True
    throttled: bool = False
    low_power_mode: bool | None = None
    detail: str = ""
    raw: dict[str, str] = field(default_factory=dict)  # raw command outputs: fixtures for the parsers


def machine_state(run: Runner = _run, system: str | None = None) -> MachineState:
    system = system or platform.system()
    if system != "Darwin":
        return MachineState(system=system, ok=True, detail="n/a (not macOS)")
    therm, pm = run(["pmset", "-g", "therm"]), run(["pmset", "-g"])
    t, lpm = parse_pmset_therm(therm), parse_lowpowermode(pm)
    problems = []
    if t["throttled"]:
        problems.append(f"thermal throttling {t['limits'] or t['warnings']}")
    if lpm:
        problems.append("Low Power Mode is on")
    return MachineState(system=system, ok=not problems, throttled=t["throttled"], low_power_mode=lpm,
                        detail="; ".join(problems) or "ok", raw={"pmset -g therm": therm, "pmset -g": pm})


def wait_until_cool(max_wait_s: float = 300.0, poll_s: float = 10.0, state_fn: Callable[[], MachineState] = machine_state,
                    sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> MachineState:
    """Block while the machine reports throttling, up to `max_wait_s`; returns the last state (caller decides
    whether a still-throttled state invalidates the measurement). Never blocks on Low Power Mode (it will not clear)."""
    deadline = clock() + max_wait_s
    st = state_fn()
    while st.throttled and clock() < deadline:
        sleep(poll_s)
        st = state_fn()
    return st
