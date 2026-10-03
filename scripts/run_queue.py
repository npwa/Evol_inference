"""Run a job queue unattended within a wall-clock budget (see evol_inference/job_queue.py), or write the default
Mac queue.

    python scripts/run_queue.py --init-m4 queue_m4.json --budget-hours 23 [--model phi3-mini ...]   # write a queue
    python scripts/run_queue.py queue_m4.json                                                        # run / resume it
    touch queue_m4.state.STOP                                                                        # stop after the current job
"""

import argparse
import json
import sys
from pathlib import Path

from evol_inference.job_queue import load_spec, run_queue


def default_m4_queue(budget_hours: float, py: str, sync: list[str] | None) -> dict:
    """Provisional order and estimates for the 24 h block. The estimates are PLACEHOLDERS to be replaced with
    per-evaluation timings from a Graviton rehearsal (plan §23); flexible jobs use whatever time remains."""
    def table(model, cfgs, share_hours, extra=()):
        return {"name": f"table-{model}", "cmd": [py, "scripts/m4_measure_table.py", "--model", model, "--configs", *cfgs,
                "--out", f"results/m4_table_{model}.jsonl", *extra],
                "est_seconds": 1800, "flexible": True, "budget_arg": "--budget-seconds", "max_seconds": int(share_hours * 3600), "min_seconds": 600}
    return {
        "budget_seconds": int(budget_hours * 3600), "margin_seconds": 900, **({"sync_cmd": sync} if sync else {}),
        "jobs": [
            {"name": "selftest", "cmd": [py, "scripts/mac_selftest.py", "--model", "phi3-mini", "--out", "results/m4_selftest.json"],
             "est_seconds": 900, "required": True},
            {"name": "thread-scan", "cmd": [py, "scripts/m4_thread_scan.py", "--model", "phi3-mini", "--out", "results/m4_thread_scan.json"],
             "est_seconds": 2400},
            table("llama3.1-8b", ["stock", "kai"], 9),
            table("phi3-mini", ["stock", "kai"], 9, ("--alphabet", "int8", "int4")),
            {"name": "analyze", "cmd": [py, "scripts/m4_analyze.py", "--table", "results/m4_table_llama3.1-8b.jsonl", "--model", "llama3.1-8b",
                                        "--out", "results/m4_analysis_llama3.1-8b.json"], "est_seconds": 300},
        ]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", nargs="?")
    ap.add_argument("--init-m4", metavar="FILE", help="write the default Mac queue to FILE and exit")
    ap.add_argument("--budget-hours", type=float, default=23.0)
    ap.add_argument("--sync-cmd", nargs="*", default=None, help="e.g. aws s3 sync results s3://bucket/run1 (run after each job)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--logs", default="results/queue_logs")
    a = ap.parse_args()
    if a.init_m4:
        Path(a.init_m4).write_text(json.dumps(default_m4_queue(a.budget_hours, sys.executable, a.sync_cmd), indent=2))
        print(f"wrote {a.init_m4}")
        return
    if not a.queue:
        ap.error("give a queue file, or --init-m4 FILE")
    state = run_queue(load_spec(a.queue), a.state or Path(a.queue).with_suffix(".state.json"), a.logs)
    print(json.dumps({k: v["status"] for k, v in state["jobs"].items()}, indent=1), "\nqueue", state["status"])
    raise SystemExit(0 if state["status"] == "finished" else 1)


if __name__ == "__main__":
    main()
