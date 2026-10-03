"""Pass/fail logic for the Mac selftest gate (scripts/mac_selftest.py), kept pure so the thresholds are unit-tested.
Thresholds come from measurements (Doc/implementation_plan_mac-m4.md §20.3, §20.4, §22):
  * KleidiAI's Q8_0 path is 8-40x less accurate than the stock path (per-row re-quantization): EXPECTED, recorded.
  * A KleidiAI Q8_0 KL divergence more than `garbage_factor` (200x) above stock is not that loss but a wrong
    kernel (the SME path under QEMU 8.2.2 was ~5,800x): FATAL for that configuration.
  * Q4_0 is bit-for-bit the same across kernel families on Graviton3: a >10% disagreement is a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Check:
    name: str
    ok: bool
    severity: str = "fatal"        # fatal: abort the run (or drop the configuration); warn: record and continue
    detail: str = ""
    data: dict = field(default_factory=dict)

    @property
    def failed_fatally(self) -> bool:
        return not self.ok and self.severity == "fatal"


def gate_platform(system: str, machine: str, allow_non_mac: bool) -> Check:
    mac = system == "Darwin" and machine == "arm64"
    if mac:
        return Check("platform", True, detail="macOS on Apple silicon")
    return Check("platform", allow_non_mac, "fatal", f"{system}/{machine} is not macOS arm64" + (" (allowed: rehearsal)" if allow_non_mac else ""))


def gate_resources(free_disk_gb: float, ram_gib: float | None, need_disk_gb: float, need_ram_gib: float) -> list[Check]:
    out = [Check("disk", free_disk_gb >= need_disk_gb, "fatal", f"{free_disk_gb:.0f} GB free, need {need_disk_gb:.0f}")]
    if ram_gib is not None:
        out.append(Check("memory", ram_gib >= need_ram_gib, "fatal", f"{ram_gib:.1f} GiB, need {need_ram_gib:.1f}"))
    return out


def gate_machine_state(ok: bool, throttled: bool, low_power_mode: bool | None, detail: str) -> list[Check]:
    return [Check("low-power-mode", not low_power_mode, "fatal", "Low Power Mode distorts speed and energy: turn it off" if low_power_mode else "off or n/a"),
            Check("thermal", not throttled, "warn", detail)]


def gate_energy(backend: str, idle_w: float, busy_w: float, min_ratio: float = 1.2, fixture_present: bool = True) -> list[Check]:
    real = backend in ("powermetrics", "rapl")
    out = [Check("energy-responds-to-load", (busy_w > idle_w * min_ratio) or not real, "fatal" if real else "warn",
                 f"{backend}: idle {idle_w:.2f} W, busy {busy_w:.2f} W" + ("" if real else " (synthetic backend: not a measurement)"),
                 {"idle_w": idle_w, "busy_w": busy_w, "backend": backend})]
    if backend == "powermetrics":
        out.append(Check("powermetrics-fixture", fixture_present, "warn",
                         "pm_fixture.plist present" if fixture_present else "no pm_fixture.plist: run `python energy_meter.py record pm_fixture.plist --seconds 10` and commit it"))
    return out


def gate_backend_is_cpu(backends: str, label: str = "") -> Check:
    bad = [b for b in ("metal", "mtl", "cuda", "vulkan") if b in backends.lower()]
    return Check(f"cpu-only[{label}]" if label else "cpu-only", not bad, "fatal",
                 f"llama.cpp backends: {backends!r}" + (f" (GPU backend active: {bad})" if bad else ""))


def gate_f16_floor(kld_f16: float, label: str, floor: float = 1e-4) -> Check:
    return Check(f"f16-floor[{label}]", abs(kld_f16) < floor, "warn", f"F16 vs the base logits: {kld_f16:.2e} nats (limit {floor:.0e})")


def gate_q8(label: str, kld_cfg: float, kld_stock: float, garbage_factor: float = 200.0) -> list[Check]:
    """Q8_0 KL divergence of configuration `label` vs the stock path."""
    ratio = kld_cfg / max(kld_stock, 1e-6)
    data = {"kld": kld_cfg, "kld_stock": kld_stock, "ratio": ratio}
    garbage = ratio > garbage_factor
    note = ("EXPECTED KleidiAI Q8_0 re-quantization loss (8-40x on Graviton3)" if 3 <= ratio <= garbage_factor
            else "no loss versus stock" if ratio < 3 else "WRONG KERNEL, not the expected loss")
    return [Check(f"q8-sane[{label}]", not garbage, "fatal", f"Q8_0 KLD {kld_cfg:.5f} vs stock {kld_stock:.5f} = x{ratio:.1f}: {note}", data)]


def gate_q4(label: str, kld_cfg: float, kld_stock: float, rel: float = 0.10) -> Check:
    dev = abs(kld_cfg - kld_stock) / max(kld_stock, 1e-9)
    return Check(f"q4-agrees[{label}]", dev <= rel, "warn", f"Q4_0 KLD {kld_cfg:.5f} vs stock {kld_stock:.5f} ({100 * dev:.1f}% apart)",
                 {"kld": kld_cfg, "kld_stock": kld_stock, "rel_dev": dev})


def gate_speed(label: str, tps: float, tps_stock: float, lo: float = 0.4, hi: float = 4.0) -> Check:
    r = tps / tps_stock if tps_stock else float("inf")
    return Check(f"speed-sane[{label}]", tps > 0 and lo <= r <= hi, "warn", f"decode {tps:.1f} tok/s = x{r:.2f} of stock", {"ratio": r})


def usable_configs(checks: list[Check], configs: list[str]) -> list[str]:
    """Configurations without a fatal failure naming them (checks carry `[label]` in their name)."""
    bad = {c for c in configs for k in checks if k.failed_fatally and k.name.endswith(f"[{c}]")}
    return [c for c in configs if c not in bad]


def overall_ok(checks: list[Check], configs: list[str], required: tuple[str, ...] = ("stock",)) -> bool:
    """Pass if no fatal check outside a configuration failed, and every `required` configuration (and at least one
    other, if any were requested) is usable."""
    global_fatal = [k for k in checks if k.failed_fatally and not any(k.name.endswith(f"[{c}]") for c in configs)]
    ok_cfg = usable_configs(checks, configs)
    return not global_fatal and all(r in ok_cfg for r in required if r in configs) and bool(ok_cfg)
