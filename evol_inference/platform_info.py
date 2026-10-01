"""Platform record stamped into every result file (Doc/implementation_plan_mac-m4.md §3.1).
Latency/energy numbers are meaningless without the machine that produced them, so a result
JSON without this block is rejected by `require_platform`.

Parsing is split from collection (`parse_*` take text) so it is unit-tested on any host
with captured `sysctl`/`/proc/cpuinfo` output; `collect()` just feeds them live data."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

PLATFORM_KEY = "platform"

# Arm feature -> (Linux /proc/cpuinfo token, macOS sysctl hw.optional.arm.* name)
ARM_FEATURES = {
    "neon": ("asimd", "AdvSIMD"),
    "dotprod": ("asimddp", "FEAT_DotProd"),
    "i8mm": ("i8mm", "FEAT_I8MM"),
    "bf16": ("bf16", "FEAT_BF16"),
    "fp16": ("fphp", "FEAT_FP16"),
    "sve": ("sve", None),
    "sve2": ("sve2", None),
    "sme": ("sme", "FEAT_SME"),
    "sme2": ("sme2", "FEAT_SME2"),
}


def parse_proc_cpuinfo(text: str) -> dict:
    """Model/feature info from Linux /proc/cpuinfo text (aarch64 or x86)."""
    feats: set[str] = set()
    model = None
    for line in text.splitlines():
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key in ("Features", "flags"):
            feats |= set(val.split())
        elif key in ("model name", "Model", "CPU part") and model is None:
            model = val
    arm = {name: tok in feats for name, (tok, _) in ARM_FEATURES.items()}
    return {"model": model, "arm_features": arm, "raw_feature_count": len(feats)}


def parse_sysctl(text: str) -> dict:
    """Model/feature/topology info from macOS `sysctl -a` (or a grep of it)."""
    kv: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([\w.]+):\s*(.*)$", line)
        if m:
            kv[m.group(1)] = m.group(2)
    arm = {}
    for name, (_, mac) in ARM_FEATURES.items():
        if mac is None:
            arm[name] = False
        else:
            arm[name] = kv.get(f"hw.optional.arm.{mac}", kv.get(f"hw.optional.{mac}")) == "1"
    topo = {}
    for lvl, label in (("0", "performance"), ("1", "efficiency")):
        v = kv.get(f"hw.perflevel{lvl}.logicalcpu")
        if v is not None:
            topo[f"{label}_cores"] = int(v)
    mem = kv.get("hw.memsize")
    return {
        "model": kv.get("machdep.cpu.brand_string"),
        "arm_features": arm,
        "core_topology": topo,
        "memory_gib": round(int(mem) / 2**30, 1) if mem else None,
    }


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def collect(llama_cpp_dir: str | os.PathLike | None = None, extra: dict | None = None) -> dict:
    """Live platform record. `llama_cpp_dir` (a git checkout) adds the commit hash;
    `extra` is merged in for run-specific facts (CMake flags, KleidiAI on/off, threads)."""
    system = platform.system()
    info: dict = {
        "system": system, "release": platform.release(), "machine": platform.machine(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(), "synthetic_host": False,
    }
    if system == "Darwin":
        info.update(parse_sysctl(_run(["sysctl", "-a"])))
        info["thermal_note"] = "record `pmset -g therm` output via extra=; Low Power Mode off"
    elif system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            info.update(parse_proc_cpuinfo(cpuinfo.read_text()))
        try:
            kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
            info["memory_gib"] = round(kb / 2**20, 1)
        except (OSError, AttributeError):
            pass
    info["is_arm"] = platform.machine().lower() in ("arm64", "aarch64")
    if llama_cpp_dir is not None:
        info["llama_cpp_commit"] = _run(["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"]).strip() or None
    if extra:
        info.update(extra)
    return info


def require_platform(result: dict) -> dict:
    """Return the result's platform block, or raise: report scripts call this first."""
    block = result.get(PLATFORM_KEY)
    if not isinstance(block, dict) or not block.get("system"):
        raise ValueError("result has no platform block; refusing to report numbers with unknown provenance")
    return block
