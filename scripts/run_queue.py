"""Run a job queue unattended within a wall-clock budget (see evol_inference/job_queue.py), or write the default
Mac queue.

    python scripts/run_queue.py --init-m4 queue_m4.json --budget-hours 22 --threads 10 --sync-cmd aws s3 sync ...   # the Mac queue
    python scripts/run_queue.py --init-rehearsal queue_rehearsal.json --limit 12                    # Graviton dress rehearsal queue
    python scripts/run_queue.py queue_m4.json                                                        # run / resume it
    touch queue_m4.state.STOP                                                                        # stop after the current job
"""

import argparse
import json
import sys
from pathlib import Path

from evol_inference.job_queue import load_spec, run_queue


def default_m4_queue(budget_hours: float, py: str, sync: list[str] | None, threads: int | None = None, llama_dir: str = "~/work/llama.cpp",
                     build_kai: str = "build-kai", build_stock: str = "build-stock", meter: str = "powermetrics",
                     share_8b_h: float = 9.0, share_phi3_h: float = 7.0, configs: tuple[str, ...] = ("stock", "kai"),
                     with_thread_scan: bool = False, prefix_configs: tuple[str, ...] = ("stock:12",)) -> dict:
    """The Mac (T4) queue. Run the selftest and the thread scan interactively first, then generate this queue with the chosen
    `threads` (Doc/m4_runbook.md). The two table jobs are flexible: each gets at most its share of the time and stops cleanly, and they
    honour the selftest verdict (an unusable configuration is dropped, a failed gate aborts). The shares are PLACEHOLDERS until the
    Graviton rehearsal's timing report (scripts/m4_timing_report.py) is available: set them from its projections."""
    common = ["--llama-dir", llama_dir, "--build-kai", build_kai, "--build-stock", build_stock, "--meter", meter,
              *(["--threads", str(threads)] if threads else [])]

    prefix = ["--prefix-config", *prefix_configs] if prefix_configs else []

    def table(model, share_h, extra=()):
        return {"name": f"table-{model}",
                "cmd": [py, "scripts/m4_measure_table.py", "--model", model, "--configs", *configs, *common, "--selftest", "results/m4_selftest.json",
                        "--out", f"results/m4_table_{model}.jsonl", *prefix, *extra],
                "est_seconds": 1800, "flexible": True, "budget_arg": "--budget-seconds", "max_seconds": int(share_h * 3600), "min_seconds": 600}

    def analyze(model, extra=()):
        return {"name": f"analyze-{model}", "est_seconds": 300, "always": True,
                "cmd": [py, "scripts/m4_analyze.py", "--table", f"results/m4_table_{model}.jsonl", "--model", model, "--out", f"results/m4_analysis_{model}.json", *extra]}

    jobs = [{"name": "selftest", "required": True, "est_seconds": 900,
             "cmd": [py, "scripts/mac_selftest.py", "--model", "phi3-mini", "--configs", "stock", "kai", "kai-nosme", *common[:6], "--meter", meter,
                     *(["--threads", str(threads)] if threads else []), "--out", "results/m4_selftest.json"]}]
    if with_thread_scan:
        jobs.append({"name": "thread-scan", "est_seconds": 2400,
                     "cmd": [py, "scripts/m4_thread_scan.py", "--model", "phi3-mini", "--configs", *configs, *common, "--out", "results/m4_thread_scan.json"]})
    jobs += [table("llama3.1-8b", share_8b_h), analyze("llama3.1-8b"),
             table("phi3-mini", share_phi3_h, ("--alphabet", "int8", "int4")), analyze("phi3-mini", ("--alphabet", "int8", "int4"))]
    return {"budget_seconds": int(budget_hours * 3600), "margin_seconds": 900, **({"sync_cmd": sync} if sync else {}), "jobs": jobs}


def rehearsal_queue(py: str, model: str, llama_dir: str, build_kai: str, build_stock: str, limit: int, budget_hours: float,
                    threads: int | None = None, scan_threads: tuple[int, ...] = (1, 8)) -> dict:
    """Queue for the Graviton dress rehearsal (Doc/graviton_rehearsal.md): the Mac workflow on real Arm Linux with a mock
    energy meter, a restricted alphabet and a small genome limit, ending in a per-evaluation timing report."""
    common = ["--llama-dir", llama_dir, "--build-kai", build_kai, "--build-stock", build_stock]
    th = ["--threads", str(threads)] if threads else []
    table = f"results/rehearsal_table_{model}.jsonl"
    return {
        "budget_seconds": int(budget_hours * 3600), "margin_seconds": 300,
        "jobs": [
            {"name": "selftest", "required": True, "est_seconds": 900,
             "cmd": [py, "scripts/mac_selftest.py", "--model", model, "--configs", "stock", "kai", *common, "--allow-non-mac", "--meter", "mock",
                     "--ctx", "512", *th, "--out", "results/rehearsal_selftest.json"]},
            {"name": "thread-scan", "est_seconds": 1200,
             "cmd": [py, "scripts/m4_thread_scan.py", "--model", model, "--configs", "stock", "kai", *common, "--threads", *map(str, scan_threads),
                     "--meter", "mock", "--out", "results/rehearsal_thread_scan.json"]},
            {"name": f"table-{model}", "flexible": True, "budget_arg": "--budget-seconds", "est_seconds": 3000, "max_seconds": 6000, "min_seconds": 600,
             "cmd": [py, "scripts/m4_measure_table.py", "--model", model, "--configs", "stock", "kai", *common, "--alphabet", "int8", "int4",
                     "--limit", str(limit), "--meter", "mock", *th, "--out", table]},
            {"name": "analyze", "est_seconds": 120, "always": True,
             "cmd": [py, "scripts/m4_analyze.py", "--table", table, "--model", model, "--out", "results/rehearsal_analysis.json"]},
            {"name": "timings", "est_seconds": 60, "always": True,
             "cmd": [py, "scripts/m4_timing_report.py", "--table", table, "--out", "results/rehearsal_timings.json"]},
        ]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", nargs="?")
    ap.add_argument("--init-m4", metavar="FILE", help="write the default Mac queue to FILE and exit")
    ap.add_argument("--init-rehearsal", metavar="FILE", help="write the Graviton rehearsal queue to FILE and exit")
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--llama-dir", default="~/llama.cpp")
    ap.add_argument("--build-kai", default="build-kai")
    ap.add_argument("--build-stock", default="build-nokai", help="the build without KleidiAI (the Graviton runbook calls it build-nokai)")
    ap.add_argument("--limit", type=int, default=12, help="rehearsal: genomes to measure")
    ap.add_argument("--meter", default="powermetrics", help="--init-m4: energy backend for the Mac queue")
    ap.add_argument("--configs", nargs="*", default=["stock", "kai"], help="--init-m4: run configurations to measure")
    ap.add_argument("--share-8b-hours", type=float, default=9.0, help="--init-m4: time cap for the Llama-3.1-8B table")
    ap.add_argument("--share-phi3-hours", type=float, default=7.0, help="--init-m4: time cap for the Phi-3 table")
    ap.add_argument("--prefix-config", nargs="*", default=["stock:12"], metavar="CFG:N",
                    help="--init-m4: measure CFG only for the first N genomes of the priority order (the reference, baselines and greedy chain): the full "
                         "table is measured in the other configurations. Default stock:12. Pass nothing to measure every configuration in full.")
    ap.add_argument("--with-thread-scan", action="store_true", help="--init-m4: include the thread scan in the queue (default: run it interactively first)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--budget-hours", type=float, default=23.0)
    ap.add_argument("--sync-cmd", nargs="*", default=None, help="e.g. aws s3 sync results s3://bucket/run1 (run after each job)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--restart-clock", type=float, default=None, metavar="HOURS",
                    help="resume with the deadline re-anchored to now + HOURS (after a long pause; the default keeps the first start's deadline)")
    ap.add_argument("--logs", default="results/queue_logs")
    a = ap.parse_args()
    if a.init_m4:
        Path(a.init_m4).write_text(json.dumps(default_m4_queue(
            a.budget_hours, sys.executable, a.sync_cmd, a.threads, a.llama_dir if a.llama_dir != "~/llama.cpp" else "~/work/llama.cpp",
            a.build_kai, a.build_stock if a.build_stock != "build-nokai" else "build-stock", a.meter, a.share_8b_hours, a.share_phi3_hours,
            tuple(a.configs), a.with_thread_scan, tuple(a.prefix_config)), indent=2))
        print(f"wrote {a.init_m4}")
        return
    if a.init_rehearsal:
        Path(a.init_rehearsal).write_text(json.dumps(rehearsal_queue(sys.executable, a.model, a.llama_dir, a.build_kai, a.build_stock, a.limit,
                                                                     a.budget_hours if a.budget_hours != 23.0 else 3.5, a.threads), indent=2))
        print(f"wrote {a.init_rehearsal}")
        return
    if not a.queue:
        ap.error("give a queue file, or --init-m4 FILE / --init-rehearsal FILE")
    state = run_queue(load_spec(a.queue), a.state or Path(a.queue).with_suffix(".state.json"), a.logs, restart_clock=a.restart_clock)
    print(json.dumps({k: v["status"] for k, v in state["jobs"].items()}, indent=1), "\nqueue", state["status"])
    raise SystemExit(0 if state["status"] == "finished" else 1)


if __name__ == "__main__":
    main()
