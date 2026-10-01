"""energy_meter.py - portable energy measurement for Evol_inference fitness evaluation.

One interface, four backends, selected by ENERGY_BACKEND or auto-detection:

  powermetrics : macOS on Apple silicon. Real on-chip CPU power. Needs passwordless sudo:
                 echo "$(whoami) ALL=(ALL) NOPASSWD: /usr/bin/powermetrics *" | sudo tee /etc/sudoers.d/powermetrics
  rapl         : Linux on Intel/AMD x86 via /sys/class/powercap. Real energy counters.
                 Usually needs root or a chmod on energy_uj.
  mock         : Any Unix. Synthetic power derived from measured CPU time plus noise.
                 Lets the GA loop run end to end on Graviton or a laptop.
  replay       : Any Unix. Feeds recorded powermetrics output through the real parser,
                 so the macOS parsing code is exercised on Linux.

Workflow:
  Linux dev:        ENERGY_BACKEND=mock python energy_meter.py selftest
  First hour on Mac: python energy_meter.py record pm_fixture.plist --seconds 10
                     python energy_meter.py selftest
  Back on Linux:    python energy_meter.py replay pm_fixture.plist   (parser regression test)
"""
from __future__ import annotations

import os
import platform
import plistlib
import random
import resource
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Keys tried in order inside the plist "processor" dict. Values in milliwatts.
# VERIFY against a real recording: `record` prints the available keys if none match.
POWER_KEYS_MW = ("cpu_power", "combined_power")
ENERGY_KEY_MJ = "cpu_energy"  # fallback: energy over the sample's elapsed_ns


@dataclass
class Measurement:
    backend: str
    duration_s: float
    energy_j: float
    samples_w: list[float] = field(default_factory=list)

    @property
    def avg_power_w(self) -> float:
        return self.energy_j / self.duration_s if self.duration_s > 0 else 0.0

    def net_energy_j(self, baseline_w: float) -> float:
        """Energy attributable to the workload after subtracting idle power."""
        return max(0.0, self.energy_j - baseline_w * self.duration_s)


class EnergyMeter(ABC):
    name = "base"

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> Measurement: ...

    def idle_baseline(self, seconds: float = 10.0) -> float:
        """Average idle power in watts. Run before each batch of candidates."""
        self.start()
        time.sleep(seconds)
        return self.stop().avg_power_w

    def measure_cmd(self, cmd: list[str], **run_kw):
        """Run a command (e.g. llama-bench) under measurement."""
        self.start()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, **run_kw)
        finally:
            m = self.stop()
        return proc, m


def _from_samples(name: str, duration_s: float, samples_w: list[float]) -> Measurement:
    avg = sum(samples_w) / len(samples_w) if samples_w else 0.0
    return Measurement(name, duration_s, avg * duration_s, samples_w)


# ---------------------------------------------------------------- powermetrics

def parse_powermetrics_sample(doc: bytes) -> float | None:
    """Parse one plist document from `powermetrics -f plist`. Returns CPU watts or None."""
    doc = doc.strip()
    if not doc:
        return None
    try:
        d = plistlib.loads(doc)
    except Exception:
        return None
    proc = d.get("processor", {})
    for key in POWER_KEYS_MW:
        if key in proc:
            return float(proc[key]) / 1000.0
    elapsed_ns = d.get("elapsed_ns")
    if ENERGY_KEY_MJ in proc and elapsed_ns:
        return (float(proc[ENERGY_KEY_MJ]) / 1000.0) / (elapsed_ns / 1e9)
    return None


def _powermetrics_cmd(interval_ms: int, sudo: bool) -> list[str]:
    return (["sudo", "-n"] if sudo else []) + [
        "powermetrics", "--samplers", "cpu_power", "-i", str(interval_ms), "-f", "plist"]


class PowermetricsMeter(EnergyMeter):
    name = "powermetrics"

    def __init__(self, interval_ms: int = 100, sudo: bool = True):
        self.interval_ms = interval_ms
        self.cmd = _powermetrics_cmd(interval_ms, sudo)

    def start(self) -> None:
        self._samples: list[float] = []
        self._buf = b""
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        time.sleep(self.interval_ms / 1000.0)  # let the first sample window open
        self._t0 = time.perf_counter()

    def _read(self) -> None:
        for chunk in iter(lambda: self._proc.stdout.read(4096), b""):
            self._buf += chunk
            while b"\0" in self._buf:  # samples are NUL-separated
                doc, self._buf = self._buf.split(b"\0", 1)
                w = parse_powermetrics_sample(doc)
                if w is not None:
                    self._samples.append(w)

    def stop(self) -> Measurement:
        duration = time.perf_counter() - self._t0
        time.sleep(self.interval_ms / 1000.0)  # flush the last window
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._reader.join(timeout=5)
        if not self._samples:
            err = self._proc.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"powermetrics produced no parsable samples. stderr: {err or '(empty)'}")
        return _from_samples(self.name, duration, self._samples)


# ------------------------------------------------------------------------ RAPL

class RaplMeter(EnergyMeter):
    name = "rapl"

    def __init__(self, domain: str = "/sys/class/powercap/intel-rapl:0"):
        self.domain = domain
        with open(f"{domain}/max_energy_range_uj") as f:
            self.max_uj = int(f.read())

    def _read_uj(self) -> int:
        with open(f"{self.domain}/energy_uj") as f:
            return int(f.read())

    def start(self) -> None:
        self._e0 = self._read_uj()
        self._t0 = time.perf_counter()

    def stop(self) -> Measurement:
        e1 = self._read_uj()
        duration = time.perf_counter() - self._t0
        delta = e1 - self._e0
        if delta < 0:  # counter wrapped
            delta += self.max_uj
        return Measurement(self.name, duration, delta / 1e6)


# ------------------------------------------------------------------------ mock

def _cpu_seconds() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


class MockMeter(EnergyMeter):
    """Synthetic model: idle_w + per_core_w * busy_cores, with multiplicative noise.
    Defaults are loosely M4-like. Only the shape matters, not the absolute numbers."""
    name = "mock"

    def __init__(self, idle_w: float = 2.0, per_core_w: float = 1.8,
                 noise: float = 0.03, seed: int | None = 0):
        self.idle_w, self.per_core_w, self.noise = idle_w, per_core_w, noise
        self._rng = random.Random(seed)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._c0 = _cpu_seconds()

    def stop(self) -> Measurement:
        duration = time.perf_counter() - self._t0
        busy_cores = (_cpu_seconds() - self._c0) / duration if duration > 0 else 0.0
        w = (self.idle_w + self.per_core_w * busy_cores) * (1 + self._rng.gauss(0, self.noise))
        return Measurement(self.name, duration, w * duration, [w])


# ---------------------------------------------------------------------- replay

class ReplayMeter(EnergyMeter):
    """Replays a recorded powermetrics fixture through the real parser."""
    name = "replay"

    def __init__(self, fixture: str, interval_s: float = 0.1):
        with open(fixture, "rb") as f:
            raw = f.read()
        self._watts = [w for w in map(parse_powermetrics_sample, raw.split(b"\0")) if w is not None]
        if not self._watts:
            raise ValueError(f"No parsable samples in {fixture}; check POWER_KEYS_MW")
        self.interval_s = interval_s
        self._i = 0

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> Measurement:
        duration = time.perf_counter() - self._t0
        k = max(1, round(duration / self.interval_s))
        samples = [self._watts[(self._i + j) % len(self._watts)] for j in range(k)]
        self._i += k
        return _from_samples(self.name, duration, samples)


# --------------------------------------------------------------------- factory

def _autodetect() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64" and shutil.which("powermetrics"):
        return "powermetrics"
    if platform.system() == "Linux" and os.access("/sys/class/powercap/intel-rapl:0/energy_uj", os.R_OK):
        return "rapl"
    return "mock"


def get_meter(backend: str | None = None, **kw) -> EnergyMeter:
    backend = backend or os.environ.get("ENERGY_BACKEND") or _autodetect()
    if backend == "replay":
        kw.setdefault("fixture", os.environ.get("ENERGY_FIXTURE", "pm_fixture.plist"))
    return {"powermetrics": PowermetricsMeter, "rapl": RaplMeter,
            "mock": MockMeter, "replay": ReplayMeter}[backend](**kw)


# ------------------------------------------------------------------------- CLI

BUSY_CMD = [sys.executable, "-c", "sum(i * i for i in range(40_000_000))"]


def _selftest(backend: str | None) -> int:
    meter = get_meter(backend)
    baseline = meter.idle_baseline(5.0)
    _, m = meter.measure_cmd(BUSY_CMD)
    ok = m.avg_power_w > baseline
    print(f"backend={meter.name} idle={baseline:.2f}W busy={m.avg_power_w:.2f}W "
          f"dur={m.duration_s:.2f}s net={m.net_energy_j(baseline):.2f}J -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _record(out: str, seconds: float, interval_ms: int = 100) -> int:
    n = max(1, int(seconds * 1000 / interval_ms))
    raw = subprocess.run(_powermetrics_cmd(interval_ms, True) + ["-n", str(n)],
                         capture_output=True, check=True).stdout
    with open(out, "wb") as f:
        f.write(raw)
    docs = [d for d in raw.split(b"\0") if d.strip()]
    parsed = [w for w in map(parse_powermetrics_sample, docs) if w is not None]
    print(f"wrote {len(docs)} samples to {out}; parsed {len(parsed)}")
    if docs and not parsed:
        d = plistlib.loads(docs[0].strip())
        print("No power key matched. Top-level keys:", sorted(d))
        print("processor keys:", sorted(d.get("processor", {})))
        return 1
    return 0


def _replay(fixture: str) -> int:
    m = ReplayMeter(fixture)
    print(f"parsed {len(m._watts)} samples, mean {sum(m._watts) / len(m._watts):.2f}W, "
          f"min {min(m._watts):.2f}W, max {max(m._watts):.2f}W")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("selftest"); t.add_argument("--backend")
    r = sub.add_parser("record"); r.add_argument("out"); r.add_argument("--seconds", type=float, default=10)
    p = sub.add_parser("replay"); p.add_argument("fixture")
    a = ap.parse_args()
    sys.exit({"selftest": lambda: _selftest(a.backend),
              "record": lambda: _record(a.out, a.seconds),
              "replay": lambda: _replay(a.fixture)}[a.cmd]())
