"""Unattended job queue with a wall-clock budget (Doc/implementation_plan_mac-m4.md §9): the Mac block is paid for
by the day, so jobs run back to back, never start one that cannot finish in the remaining time, hand the leftover
time to jobs that can use it, and survive being restarted (state is on disk).

Queue spec (JSON):
  {"budget_seconds": 82800, "margin_seconds": 600, "sync_cmd": ["aws","s3","sync","results","s3://bucket/run1"],
   "jobs": [{"name": "selftest", "cmd": ["python", "scripts/mac_selftest.py"], "est_seconds": 600, "required": true},
            {"name": "table", "cmd": [...], "est_seconds": 3600, "flexible": true, "budget_arg": "--budget-seconds",
             "max_seconds": 36000}]}
 * required: if it fails the queue stops (e.g. the selftest gate);
 * est_seconds: a job whose estimate exceeds the time left (minus the margin) is skipped, unless it is flexible;
 * flexible: given min(time left - margin, max_seconds) through `budget_arg` appended to its command, and expected to stop
   cleanly by itself (scripts/m4_measure_table.py does); a hard timeout of that budget + grace still applies;
 * always: cheap post-processing (analysis, timing report) that runs even when the deadline has passed: it is the point of the run;
 * a stop file (<state>.STOP) is honoured between jobs and also passed to jobs in $QUEUE_STOP_FILE;
 * restart_clock (hours): re-anchor the deadline to now + hours when resuming after a long pause: the deadline is otherwise fixed
   at first start, so time spent idle between attempts (found in the first Graviton rehearsal: 2 h) would eat the budget.
State: <state>.json with each job's status, return code, times; the deadline is fixed at first start so a restart
keeps the same budget.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

GRACE_SECONDS = 120


def load_spec(path: str | Path) -> dict:
    spec = json.loads(Path(path).read_text())
    names = [j["name"] for j in spec["jobs"]]
    if len(set(names)) != len(names):
        raise ValueError("job names must be unique")
    for j in spec["jobs"]:
        if j.get("flexible") and not j.get("budget_arg"):
            raise ValueError(f"flexible job {j['name']!r} needs budget_arg")
    return spec


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def run_subprocess(cmd: list[str], log: Path, timeout: float, env: dict) -> int:
    """Run `cmd` in its own process group, logging to `log`; kill the group on timeout. Returns the return code
    (124 on timeout)."""
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            return 124


def run_queue(spec: dict, state_path: str | Path, log_dir: str | Path, now: Callable[[], float] = time.time,
              run: Callable[[list[str], Path, float, dict], int] = run_subprocess,
              sync: Callable[[list[str]], int] | None = None, echo: Callable[[str], None] = print,
              restart_clock: float | None = None) -> dict:
    state_path, log_dir = Path(state_path), Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path)
    state.setdefault("jobs", {})
    state.setdefault("started", now())
    state.setdefault("deadline", state["started"] + spec["budget_seconds"])
    if restart_clock is not None:
        state["deadline"] = now() + restart_clock * 3600
        state["restarted"] = now()
        echo(f"clock restarted: {restart_clock:g} h from now")
    margin = spec.get("margin_seconds", 600)
    stop_file = state_path.with_suffix(".STOP")
    sync_cmd = spec.get("sync_cmd")
    sync = sync or (lambda c: subprocess.run(c, capture_output=True).returncode)
    state["status"] = "running"

    for job in spec["jobs"]:
        name, st = job["name"], state["jobs"].setdefault(job["name"], {})
        if st.get("status") == "done":
            continue
        if stop_file.exists():
            state["status"] = "stopped"
            echo(f"stop file found before {name}")
            break
        left = state["deadline"] - now() - margin
        cmd, timeout = list(job["cmd"]), None
        if job.get("flexible"):
            give = min(left, job.get("max_seconds", left))
            if give < job.get("min_seconds", 60):
                st.update(status="skipped_budget", reason=f"only {left:.0f}s left")
                echo(f"skip {name}: {left:.0f}s left")
                _save(state_path, state)
                continue
            cmd += [job["budget_arg"], str(int(give))]
            timeout = give + GRACE_SECONDS
        elif job.get("est_seconds", 0) > left and not job.get("always"):
            st.update(status="skipped_budget", reason=f"needs {job['est_seconds']}s, {left:.0f}s left")
            echo(f"skip {name}: needs {job['est_seconds']}s, {left:.0f}s left")
            _save(state_path, state)
            continue
        else:
            timeout = max(job.get("timeout_seconds", job.get("est_seconds", 3600) * 3), 1)
        st.update(status="running", start=now(), cmd=cmd, log=str(log_dir / f"{name}.log"))
        _save(state_path, state)
        echo(f"run {name} (timeout {timeout:.0f}s)")
        rc = run(cmd, log_dir / f"{name}.log", timeout, dict(os.environ, QUEUE_STOP_FILE=str(stop_file)))
        st.update(status="done" if rc == 0 else ("timeout" if rc == 124 else "failed"), rc=rc, end=now())
        _save(state_path, state)
        echo(f"  {name}: {st['status']} (rc {rc}, {st['end'] - st['start']:.0f}s)")
        if sync_cmd:
            st["sync_rc"] = sync(sync_cmd)  # best effort: a failed sync never stops the queue
            _save(state_path, state)
        if rc != 0 and job.get("required"):
            state["status"] = "aborted"
            echo(f"required job {name} failed: queue aborted")
            break
    else:
        state["status"] = "finished"
    state["ended"] = now()
    _save(state_path, state)
    return state
