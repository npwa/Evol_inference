"""Per-genome measurement for the Mac (T4) day (Doc/implementation_plan_mac-m4.md §9, §22.6): for one genome
and one run configuration, measure accuracy (KL divergence vs the original model), decode and prefill speed and
energy per token, and record the machine state.

Design (why a *table*, not a GA, on the Mac): one evaluation costs roughly a minute or more on Arm (a 2048-token KL
run plus a speed run plus an energy window), so a 600-evaluation GA per model and configuration does not fit in
one 24 h block. Measuring every genome of the (small) alphabet once, interleaved across configurations, gives the
true front on the real hardware, and the GA / greedy comparison is then run offline on the table
(`measured_table.py`, `scripts/m4_analyze.py`). `priority_order` makes a time-limited run still useful: baselines
first, then the sensitivity-greedy chain, then the rest in a seeded random order (any prefix is a random sample).

Everything here runs on any host with llama.cpp builds (the desktop rehearsal uses x86 builds, a proxy model and the
mock meter); results measured with a mock/replay meter are stamped `synthetic`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from evol_inference.genome import Genome, N_SUPER_BLOCKS, Precision
from evol_inference.gguf_assembler import AssembledGgufProvider, GgufAssembler
from evol_inference.llama_probes import LlamaBenchProbe, LlamaKldProbe
from evol_inference.mac_env import MachineState, machine_state, wait_until_cool
from evol_inference.model_spec import ModelSpec
from evol_inference.sensitivity import SensitivityTable
from evol_inference.tabulated import enumerate_genomes, genome_code


@dataclass(frozen=True)
class RunConfig:
    """One way of running llama.cpp: which build, extra flags per tool, extra environment."""
    label: str
    build: str                       # build directory (contains bin/llama-perplexity, bin/llama-bench)
    ppl_args: tuple[str, ...] = ()   # e.g. ("-nr",)
    bench_args: tuple[str, ...] = () # e.g. ("--repack", "0")
    env: tuple[tuple[str, str], ...] = ()


def standard_configs(build_kai: str, build_stock: str | None) -> dict[str, RunConfig]:
    """The configurations the plan compares. `stock` (a build without KleidiAI, which keeps ggml's own Arm
    repack path) is the fair speed baseline; `kai-nr` (no repack at all) is NOT, but is kept for the A/B."""
    cfg = {
        "kai": RunConfig("kai", build_kai),
        "kai-nosme": RunConfig("kai-nosme", build_kai, env=(("GGML_KLEIDIAI_SME", "0"),)),
        "kai-nr": RunConfig("kai-nr", build_kai, ppl_args=("-nr",), bench_args=("--repack", "0")),
    }
    if build_stock:
        cfg["stock"] = RunConfig("stock", build_stock)
    return cfg


# ------------------------------------------------------------------ ordering and persistence

def greedy_chain(spec: ModelSpec, table: SensitivityTable) -> list[Genome]:
    """Genomes with the k most sensitive blocks at the highest precision and the rest at the lowest, k = 0..8."""
    levels = sorted(spec.alphabet, key=lambda p: -p.bits)
    hi, lo = levels[0], levels[-1]
    order = table.ranking(lo)
    return [[hi if i in order[:k] else lo for i in range(N_SUPER_BLOCKS)] for k in range(N_SUPER_BLOCKS + 1)]


def priority_order(spec: ModelSpec, alphabet: Sequence[Precision] | None = None, table: SensitivityTable | None = None,
                   seed: int = 0, include_reference: bool = True) -> list[Genome]:
    """Measurement order. The model's reference genome (all-F16, or all-Q8_0 for the 8B model) comes first even when it
    lies outside a restricted alphabet: every gain and penalty is relative to it, so an analysis without it has no
    baseline (found in the rehearsal, where a Q8_0/Q4_0 table had no F16 row)."""
    alphabet = tuple(alphabet or spec.alphabet)
    first = [spec.reference_genome()] if include_reference else []
    first += [g for g in spec.baselines().values() if set(g) <= set(alphabet)]
    if table is not None:
        first += [g for g in greedy_chain(spec, table) if set(g) <= set(alphabet)]
    rest = [g for g in enumerate_genomes(alphabet)]
    random.Random(seed).shuffle(rest)
    seen, out = set(), []
    for g in first + rest:
        c = genome_code(g)
        if c not in seen:
            seen.add(c)
            out.append(g)
    return out


def parse_prefix_limits(items: Sequence[str] | None) -> dict[str, int]:
    """['stock:12'] -> {'stock': 12}: measure that configuration only for the first N genomes of the priority order."""
    out: dict[str, int] = {}
    for it in items or ():
        label, _, n = it.partition(":")
        if not label or not n.isdigit():
            raise ValueError(f"--prefix-config expects CONFIG:N, got {it!r}")
        out[label] = int(n)
    return out


def configs_for(index: int, labels: Sequence[str], limits: dict[str, int]) -> list[str]:
    """Configurations to measure for the genome at `index` in the priority order."""
    return [c for c in labels if index < limits.get(c, 10**9)]


def should_stop(elapsed_s: float, budget_s: float | None, recent_s: Sequence[float], next_has_f16: bool = False,
                safety: float = 1.15) -> bool:
    """Stop BEFORE starting a genome that would overrun the budget, using the median cost of the recent quantized genomes
    (3x for a genome with F16 blocks). A flexible queue job is killed at budget + grace, which loses the genome in flight and
    marks the job timed out (first Graviton rehearsal); finishing early avoids both. With no history, only the plain
    budget check applies."""
    if budget_s is None:
        return False
    if elapsed_s > budget_s:
        return True
    if not recent_s:
        return False
    est = sorted(recent_s)[len(recent_s) // 2] * (3.0 if next_has_f16 else 1.0)
    return elapsed_s + safety * est > budget_s


def append_row(path: str | Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def done_keys(rows: Sequence[dict]) -> set[tuple[str, str]]:
    return {(r["genome"], r["config"]) for r in rows if "error" not in r}


def assembly_space_gb(sources: dict, precisions) -> float:
    """Upper bound on the size of any assembled genome file over these precisions: the size of the largest source among
    them (a mixed file is never larger than the all-largest-precision file). Used to check the scratch disk up front."""
    sizes = [Path(sources[p]).stat().st_size for p in precisions if p in sources]
    if not sizes:
        raise ValueError("no source GGUF for the requested precisions")
    return max(sizes) / 1e9


# ------------------------------------------------------------------ the measurer

@dataclass
class MeasureSettings:
    threads: int
    ctx: int = 2048
    chunks: int = 1
    n_prompt: int = 512
    n_gen: int = 128
    repeats: int = 3
    energy_method: str = "total"
    min_energy_window_s: float = 20.0   # shorter windows give too few powermetrics samples: flagged in the row
    cooldown_wait_s: float = 300.0
    ngl: int = 0                        # CPU only: Metal / GPU offload must stay off on the Mac


class GenomeMeasurer:
    """Measures genomes under several run configurations, sharing one assembled GGUF and one set of base logits."""

    def __init__(self, spec: ModelSpec, sources: dict[Precision, Path], llama_dir: Path, configs: Sequence[RunConfig],
                 text_file: Path, work_dir: Path, meter, settings: MeasureSettings, idle_s: float = 10.0,
                 state_fn: Callable[[], MachineState] = machine_state):
        self.spec, self.meter, self.s, self.state_fn = spec, meter, settings, state_fn
        self.configs = {c.label: c for c in configs}
        work_dir.mkdir(parents=True, exist_ok=True)
        self.provider = AssembledGgufProvider(GgufAssembler(spec, sources), work_dir / f"work_{spec.key}.gguf")
        self.base = work_dir / f"base_{spec.key}_{spec.reference_precision.value}_c{settings.ctx}_n{settings.chunks}.kld"
        base_model = sources[spec.reference_precision]
        self.kld, self.bench = {}, {}
        self.idle_s = idle_s
        self.idle_w = meter.idle_baseline(idle_s) if idle_s > 0 else 0.0
        for c in configs:
            b = llama_dir / c.build / "bin"
            self.kld[c.label] = LlamaKldProbe(b / "llama-perplexity", text_file, self.provider, self.base, base_model,
                                              ctx=settings.ctx, chunks=settings.chunks, threads=settings.threads,
                                              n_gpu_layers=settings.ngl, extra_args=c.ppl_args, env=dict(c.env))
            self.bench[c.label] = LlamaBenchProbe(b / "llama-bench", self.provider, meter, self.idle_w, settings.threads,
                                                  n_prompt=settings.n_prompt, n_gen=settings.n_gen, repeats=settings.repeats,
                                                  energy_method=settings.energy_method, n_gpu_layers=settings.ngl,
                                                  extra_args=c.bench_args, env=dict(c.env))
        self.base_config = "stock" if "stock" in self.configs else next(iter(self.configs))
        self.ppl0: float | None = None

    def ensure_base(self) -> float:
        """Save the original model's logits once (with the base configuration, preferring the build without
        KleidiAI) and share its perplexity with the other configurations' probes."""
        probe = self.kld[self.base_config]
        self.ppl0 = probe.ensure_base()
        for p in self.kld.values():
            p.ppl0 = self.ppl0
        return self.ppl0

    def refresh_idle(self) -> float:
        self.idle_w = self.meter.idle_baseline(self.idle_s) if self.idle_s > 0 else 0.0
        for b in self.bench.values():
            b.idle_w = self.idle_w
        return self.idle_w

    def measure(self, genome: Genome, config: str) -> dict:
        if self.ppl0 is None:
            self.ensure_base()
        t0 = time.time()
        state = wait_until_cool(self.s.cooldown_wait_s, state_fn=self.state_fn) if self.s.cooldown_wait_s > 0 else self.state_fn()
        row = {"genome": genome_code(genome), "config": config, "model": self.spec.key, "t_start": t0, "base_ppl": self.ppl0,
               "threads": self.s.threads, "ctx": self.s.ctx, "idle_w": self.idle_w, "energy_backend": self.meter.name,
               "machine_ok": state.ok, "machine_detail": state.detail}
        k = self.kld[config].measure(genome)
        row.update(kld=k.mean_kld, kld_stderr=k.kld_stderr, same_top_p=k.same_top_p)
        t1 = time.time()
        se = self.bench[config].measure(genome)
        window = time.time() - t1
        row.update(decode_tps=se.decode_tps, prefill_tps=se.prefill_tps, decode_spread=se.decode_tps_spread,
                   joules_per_token=se.joules_per_token, avg_power_w=se.avg_power_w, bench_wall_s=window,
                   synthetic=se.synthetic,
                   short_energy_window=window < self.s.min_energy_window_s)
        row["wall_s"] = time.time() - t0
        return row
