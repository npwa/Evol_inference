import json
import sys
import time

import pytest

from evol_inference.job_queue import load_spec, run_queue, run_subprocess


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def fake_run(clock, durations, rcs=None, log=None):
    """Runner that advances the fake clock by the job's duration instead of running anything."""
    def run(cmd, logfile, timeout, env):
        log.append((list(cmd), timeout)) if log is not None else None
        name = cmd[1]
        clock.t += durations.get(name, 10)
        return (rcs or {}).get(name, 0)
    return run


def spec(jobs, budget=3600, margin=100, **kw):
    return {"budget_seconds": budget, "margin_seconds": margin, "jobs": jobs, **kw}


def job(name, est=100, **kw):
    return {"name": name, "cmd": ["x", name], "est_seconds": est, **kw}


def test_runs_jobs_in_order_and_records_state(tmp_path):
    c, calls = Clock(), []
    st = run_queue(spec([job("a"), job("b")]), tmp_path / "s.json", tmp_path / "logs", now=c, run=fake_run(c, {"a": 50, "b": 70}, log=calls), echo=lambda *_: None)
    assert [x[0][1] for x in calls] == ["a", "b"] and st["status"] == "finished"
    assert st["jobs"]["a"]["status"] == "done" and st["jobs"]["b"]["end"] - st["jobs"]["b"]["start"] == 70
    assert json.loads((tmp_path / "s.json").read_text())["status"] == "finished"


def test_job_that_cannot_finish_in_the_time_left_is_skipped_not_started(tmp_path):
    c, calls = Clock(), []
    jobs = [job("a", est=100), job("big", est=5000), job("small", est=100)]
    st = run_queue(spec(jobs, budget=1000, margin=100), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {}, log=calls), echo=lambda *_: None)
    assert st["jobs"]["big"]["status"] == "skipped_budget" and st["jobs"]["small"]["status"] == "done"
    assert [x[0][1] for x in calls] == ["a", "small"]


def test_flexible_job_gets_the_remaining_time_capped_by_max_seconds(tmp_path):
    c, calls = Clock(), []
    flex = job("t", flexible=True, budget_arg="--budget-seconds", max_seconds=500, min_seconds=60)
    run_queue(spec([job("a"), flex], budget=2000, margin=100), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {"a": 200}, log=calls), echo=lambda *_: None)
    cmd, timeout = calls[1]
    assert cmd[-2:] == ["--budget-seconds", "500"] and timeout == 500 + 120          # capped by max_seconds
    c2, calls2 = Clock(), []
    flex2 = job("t", flexible=True, budget_arg="--budget-seconds", min_seconds=60)
    run_queue(spec([job("a"), flex2], budget=2000, margin=100), tmp_path / "s2.json", tmp_path / "l2", now=c2, run=fake_run(c2, {"a": 200}, log=calls2), echo=lambda *_: None)
    assert calls2[1][0][-1] == str(2000 - 200 - 100)                                  # all that is left, minus the margin


def test_flexible_job_is_skipped_when_too_little_time_is_left(tmp_path):
    c = Clock()
    flex = job("t", flexible=True, budget_arg="--budget-seconds", min_seconds=300)
    st = run_queue(spec([job("a"), flex], budget=500, margin=100), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {"a": 150}), echo=lambda *_: None)
    assert st["jobs"]["t"]["status"] == "skipped_budget"


def test_required_job_failure_aborts_the_queue_but_optional_failure_does_not(tmp_path):
    c, calls = Clock(), []
    st = run_queue(spec([job("gate", required=True), job("b")]), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {}, {"gate": 1}, calls), echo=lambda *_: None)
    assert st["status"] == "aborted" and [x[0][1] for x in calls] == ["gate"]
    c2, calls2 = Clock(), []
    st2 = run_queue(spec([job("opt"), job("b")]), tmp_path / "s2.json", tmp_path / "l2", now=c2, run=fake_run(c2, {}, {"opt": 3}, calls2), echo=lambda *_: None)
    assert st2["jobs"]["opt"]["status"] == "failed" and st2["jobs"]["b"]["status"] == "done" and st2["status"] == "finished"


def test_resume_skips_done_jobs_and_keeps_the_original_deadline(tmp_path):
    c, calls = Clock(), []
    jobs = [job("a"), job("b", est=100)]
    run_queue(spec(jobs, budget=1000), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {"a": 10}, {"b": 1}, calls), echo=lambda *_: None)
    deadline = json.loads((tmp_path / "s.json").read_text())["deadline"]
    c.t += 5000                                                    # the restart happens long after the budget
    calls.clear()
    st = run_queue(spec(jobs, budget=1000), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {}, {}, calls), echo=lambda *_: None)
    assert st["deadline"] == deadline and calls == []              # a is done; b is failed-then-retried but the deadline has passed
    assert st["jobs"]["b"]["status"] == "skipped_budget"


def test_stop_file_stops_between_jobs(tmp_path):
    c, calls = Clock(), []
    (tmp_path / "s.STOP").write_text("")
    st = run_queue(spec([job("a")]), tmp_path / "s.json", tmp_path / "l", now=c, run=fake_run(c, {}, {}, calls), echo=lambda *_: None)
    assert st["status"] == "stopped" and calls == []


def test_sync_runs_after_each_job_and_its_failure_is_not_fatal(tmp_path):
    c, synced = Clock(), []
    st = run_queue(spec([job("a"), job("b")], sync_cmd=["sync", "now"]), tmp_path / "s.json", tmp_path / "l", now=c,
                   run=fake_run(c, {}), sync=lambda cmd: (synced.append(cmd), 7)[1], echo=lambda *_: None)
    assert len(synced) == 2 and st["status"] == "finished" and st["jobs"]["a"]["sync_rc"] == 7


def test_load_spec_validation(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(spec([job("a"), job("a")])))
    with pytest.raises(ValueError, match="unique"):
        load_spec(p)
    p.write_text(json.dumps(spec([job("a", flexible=True)])))
    with pytest.raises(ValueError, match="budget_arg"):
        load_spec(p)


def test_real_subprocess_runner_logs_returns_code_and_kills_on_timeout(tmp_path):
    log = tmp_path / "j.log"
    assert run_subprocess([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], log, 30, {}) == 3
    assert "hello" in log.read_text()
    t0 = time.time()
    assert run_subprocess([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path / "k.log", 1, {}) == 124
    assert time.time() - t0 < 20                                   # killed, did not wait the 60 s


def test_default_m4_queue_is_valid_and_its_flexible_jobs_use_a_flag_the_script_accepts(tmp_path):
    import os
    import subprocess

    q = tmp_path / "q.json"
    subprocess.run([sys.executable, "scripts/run_queue.py", "--init-m4", str(q), "--budget-hours", "23", "--sync-cmd", "echo", "sync"],
                   check=True, capture_output=True, env=dict(os.environ, PYTHONPATH="."))
    spec = load_spec(q)
    assert spec["budget_seconds"] == 23 * 3600 and spec["sync_cmd"] == ["echo", "sync"]
    assert spec["jobs"][0]["name"] == "selftest" and spec["jobs"][0]["required"]       # the gate runs first and aborts the queue
    flexible = [j for j in spec["jobs"] if j.get("flexible")]
    assert flexible and all(j["budget_arg"] == "--budget-seconds" for j in flexible)
    script = open("scripts/m4_measure_table.py").read()
    assert all(("ap.add_argument(\"%s\"" % j["budget_arg"]) in script for j in flexible)  # the flag really exists
    for j in spec["jobs"]:                                                                # every referenced script exists
        assert any(part.startswith("scripts/") and os.path.exists(part) for part in j["cmd"]), j["name"]


def test_rehearsal_queue_is_valid_and_ordered(tmp_path):
    import os
    import subprocess

    q = tmp_path / "r.json"
    subprocess.run([sys.executable, "scripts/run_queue.py", "--init-rehearsal", str(q), "--limit", "12", "--threads", "8"],
                   check=True, capture_output=True, env=dict(os.environ, PYTHONPATH="."))
    spec = load_spec(q)
    names = [j["name"] for j in spec["jobs"]]
    assert names == ["selftest", "thread-scan", "table-phi3-mini", "analyze", "timings"] and spec["jobs"][0]["required"]
    table = spec["jobs"][2]
    assert table["flexible"] and table["cmd"][table["cmd"].index("--limit") + 1] == "12" and "--alphabet" in table["cmd"]
    assert all("build-nokai" in j["cmd"] for j in spec["jobs"][:3])           # the Graviton runbook's name for the no-KleidiAI build
    assert "mock" in table["cmd"] and spec["budget_seconds"] == int(3.5 * 3600)
    for j in spec["jobs"]:
        assert any(part.startswith("scripts/") and os.path.exists(part) for part in j["cmd"]), j["name"]
