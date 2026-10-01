"""Probes that drive llama.cpp binaries (Doc/implementation_plan_mac-m4.md §3.5, §8.5):
`llama-perplexity` for accuracy and `llama-bench` for prefill/decode speed, the latter
under an `EnergyMeter` so speed and energy come from the same run.

All process execution goes through an injectable `runner`, so parsing and orchestration
are unit-tested at tier T0 with captured output; the real binaries are exercised from
tier T1 on. Output formats are llama.cpp's and can drift between versions -- the parsers
raise loudly (never return a guess) and each has a captured-output fixture test.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from evol_inference.genome import Genome
from evol_inference.objectives import SpeedEnergy

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(cmd: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(cmd), capture_output=True, text=True)


class ProbeError(RuntimeError):
    pass


# ------------------------------------------------------------------ perplexity

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")


def parse_perplexity(text: str) -> float:
    """PPL from `llama-perplexity` output ('Final estimate: PPL = 4.3708 +/- 0.1'). The
    final estimate is used, not intermediate '[n]x.xx' chunk values."""
    m = _PPL_RE.findall(text)
    if not m:
        raise ProbeError("no 'Final estimate: PPL = ...' line in llama-perplexity output")
    return float(m[-1])


class LlamaPerplexityProbe:
    """Accuracy probe: perplexity of the assembled GGUF over a fixed text file (the
    WikiText-2 test text, `eval_data.write_wikitext2_test`), `--ctx-size` tokens (default
    2048) per chunk, `chunks` chunks (default 1: one forward pass, deterministic). Note
    llama-perplexity scores only the second half of each chunk -- see `eval_data`."""
    synthetic = False

    def __init__(self, perplexity_bin: str | Path, text_file: str | Path,
                 gguf_provider: Callable[[Genome], Path], ctx: int = 2048, chunks: int = 1,
                 threads: int | None = None, n_gpu_layers: int = 0, runner: Runner = _default_runner):
        self.bin, self.text, self.provider = str(perplexity_bin), str(text_file), gguf_provider
        self.ctx, self.chunks, self.threads, self.ngl, self.runner = ctx, chunks, threads, n_gpu_layers, runner

    def command(self, gguf_path: Path) -> list[str]:
        cmd = [self.bin, "-m", str(gguf_path), "-f", self.text, "-c", str(self.ctx),
               "--chunks", str(self.chunks), "-ngl", str(self.ngl)]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        return cmd

    def perplexity(self, genome: Genome) -> float:
        proc = self.runner(self.command(self.provider(genome)))
        if proc.returncode != 0:
            raise ProbeError(f"llama-perplexity failed ({proc.returncode}): {proc.stderr[-500:]}")
        return parse_perplexity(proc.stdout + "\n" + proc.stderr)  # llama.cpp logs to stderr


# ------------------------------------------------------------------------- KLD

@dataclass(frozen=True)
class KldResult:
    mean_kld: float          # nats/token vs the reference model's logits
    kld_stderr: float
    ppl_ratio: float         # PPL(Q)/PPL(base) -- far noisier than KLD on small samples
    ppl_ratio_stderr: float
    rms_dp: float            # percent
    same_top_p: float        # percent


_KLD_RE = re.compile(r"Mean\s+KLD:\s*(-?[0-9.eE+-]+)\s*\S*\s*([0-9.eE+-]+)")
_RATIO_RE = re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.eE+-]+)\s*\S*\s*([0-9.eE+-]+)")
_RMSDP_RE = re.compile(r"RMS\s+.p\s*:\s*([0-9.eE+-]+)")
_TOP_RE = re.compile(r"Same top p:\s*([0-9.eE+-]+)")


def parse_kld(text: str) -> KldResult:
    """Statistics from `llama-perplexity --kl-divergence` output. Raises if the KLD line is
    missing; never returns a guess."""
    m = _KLD_RE.search(text)
    if m is None:
        raise ProbeError("no 'Mean    KLD:' line in llama-perplexity --kl-divergence output")
    r = _RATIO_RE.search(text)
    d = _RMSDP_RE.search(text)
    t = _TOP_RE.search(text)
    return KldResult(
        float(m.group(1)), float(m.group(2)),
        float(r.group(1)) if r else float("nan"), float(r.group(2)) if r else float("nan"),
        float(d.group(1)) if d else float("nan"), float(t.group(1)) if t else float("nan"),
    )


class LlamaKldProbe:
    """Accuracy probe on KL divergence to the logits of a *base model file* (D6: the low-noise
    replacement for raw perplexity deltas, which are dominated by sample noise at 8-bit; see
    plan §16 step 5). `perplexity(genome)` returns `ppl0 * exp(KLD)`, so the evaluator's penalty
    `(ppl - ppl_ref) / ppl_ref` is `exp(KLD - KLD_ref) - 1`: the fractional perplexity increase the
    quantization is expected to cause, in the base project's units, independent of which
    tokens happened to be sampled.

    The base is the *original* model (the F16 GGUF; Q8_0 for models whose F16 does not fit), not
    an assembled genome: the tensors the genome does not control (embeddings, output head) are
    quantized in every genome, and measuring against the true model keeps that constant cost
    visible instead of silently folding it into the reference (it is about half of uniform
    Q8_0's KLD on Phi-3). Base logits are saved once to `base_logits`, specific to (base model,
    text, ctx, chunks, build) -- encode those in the file name."""
    synthetic = False

    def __init__(self, perplexity_bin: str | Path, text_file: str | Path,
                 gguf_provider: Callable[[Genome], Path], base_logits: str | Path,
                 base_model: str | Path, ctx: int = 2048, chunks: int = 1,
                 threads: int | None = None, n_gpu_layers: int = 0, runner: Runner = _default_runner):
        self.bin, self.text, self.provider = str(perplexity_bin), str(text_file), gguf_provider
        self.base, self.base_model = Path(base_logits), Path(base_model)
        self.ctx, self.chunks, self.threads, self.ngl, self.runner = ctx, chunks, threads, n_gpu_layers, runner
        self.ppl0: float | None = None
        self.history: dict[tuple[str, ...], KldResult] = {}

    def _cmd(self, gguf_path: Path, *extra: str) -> list[str]:
        cmd = [self.bin, "-m", str(gguf_path), "-f", self.text, "-c", str(self.ctx),
               "--chunks", str(self.chunks), "-ngl", str(self.ngl), *extra]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        return cmd

    def _run(self, cmd) -> str:
        proc = self.runner(cmd)
        if proc.returncode != 0:
            raise ProbeError(f"llama-perplexity failed ({proc.returncode}): {proc.stderr[-500:]}")
        return proc.stdout + "\n" + proc.stderr

    def ensure_base(self) -> float:
        """Save the base model's logits (if absent) and return the base perplexity ppl0."""
        if not self.base.exists():
            out = self._run(self._cmd(self.base_model, "--kl-divergence-base", str(self.base)))
        else:  # logits exist from an earlier session: still need ppl0
            out = self._run(self._cmd(self.base_model))
        self.ppl0 = parse_perplexity(out)
        return self.ppl0

    def measure_file(self, gguf_path: Path) -> KldResult:
        if self.ppl0 is None:
            self.ensure_base()
        out = self._run(self._cmd(gguf_path, "--kl-divergence-base", str(self.base), "--kl-divergence"))
        return parse_kld(out)

    def measure(self, genome: Genome) -> KldResult:
        res = self.measure_file(self.provider(genome))
        self.history[tuple(p.value for p in genome)] = res
        return res

    def perplexity(self, genome: Genome) -> float:
        import math

        kld = self.measure(genome).mean_kld  # also establishes ppl0
        return self.ppl0 * math.exp(kld)


# ----------------------------------------------------------------------- bench

@dataclass(frozen=True)
class BenchRow:
    n_prompt: int
    n_gen: int
    avg_ts: float
    stddev_ts: float
    samples_ts: tuple[float, ...]


def parse_llama_bench_json(text: str) -> list[BenchRow]:
    """Rows from `llama-bench -o json`. Keys used: n_prompt, n_gen, avg_ts, stddev_ts and
    (when present) samples_ts. Verify against the pinned llama.cpp build -- see the fixture."""
    try:
        start = text.index("[")
        data = json.loads(text[start:text.rindex("]") + 1])
    except ValueError as e:
        raise ProbeError(f"llama-bench output is not JSON: {e}") from e
    rows = []
    for d in data:
        try:
            rows.append(BenchRow(int(d["n_prompt"]), int(d["n_gen"]), float(d["avg_ts"]),
                                 float(d.get("stddev_ts", 0.0)),
                                 tuple(float(x) for x in d.get("samples_ts", [d["avg_ts"]]))))
        except KeyError as e:
            raise ProbeError(f"llama-bench row missing key {e}: {d}") from e
    if not rows:
        raise ProbeError("llama-bench returned no rows")
    return rows


def _mad_rel(xs: Sequence[float]) -> float:
    med = statistics.median(xs)
    return statistics.median(abs(x - med) for x in xs) / med if med else 0.0


class LlamaBenchProbe:
    """Speed + energy probe. Prefill and decode are separate `llama-bench` invocations (so
    each can be given its own energy window); decode energy is what the search uses (D1).

    energy_method:
      "total"        net joules over the whole decode run / generated tokens. Includes
                     model-load and warmup energy, a constant that biases small runs.
      "differential" two runs with different repeat counts; (E_hi - E_lo) / extra tokens.
                     Cancels the constant overhead, at the cost of a second run and noisier
                     subtraction. Use on the target to compare against "total" (plan §8).
    """
    synthetic = False

    def __init__(self, bench_bin: str | Path, gguf_provider: Callable[[Genome], Path], meter,
                 idle_w: float, threads: int, n_prompt: int = 512, n_gen: int = 128,
                 repeats: int = 3, energy_method: str = "total", n_gpu_layers: int = 0,
                 runner: Callable | None = None):
        if energy_method not in ("total", "differential"):
            raise ValueError("energy_method must be 'total' or 'differential'")
        self.bin, self.provider, self.meter = str(bench_bin), gguf_provider, meter
        self.idle_w, self.threads = idle_w, threads
        self.n_prompt, self.n_gen, self.repeats = n_prompt, n_gen, repeats
        self.energy_method, self.ngl = energy_method, n_gpu_layers
        self.runner = runner  # optional replacement for meter.measure_cmd (tests)

    def command(self, gguf: Path, n_prompt: int, n_gen: int, repeats: int) -> list[str]:
        return [self.bin, "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_gen),
                "-r", str(repeats), "-t", str(self.threads), "-ngl", str(self.ngl), "-o", "json"]

    def _run(self, cmd: list[str]):
        proc, m = (self.runner or self.meter.measure_cmd)(cmd)
        if proc.returncode != 0:
            raise ProbeError(f"llama-bench failed ({proc.returncode}): {proc.stderr[-500:]}")
        return parse_llama_bench_json(proc.stdout), m

    def measure(self, genome: Genome) -> SpeedEnergy:
        gguf = self.provider(genome)
        pp_rows, _ = self._run(self.command(gguf, self.n_prompt, 0, self.repeats))
        tg_rows, m_hi = self._run(self.command(gguf, 0, self.n_gen, self.repeats))
        pp = next(r for r in pp_rows if r.n_prompt > 0)
        tg = next(r for r in tg_rows if r.n_gen > 0)
        if self.energy_method == "total":
            net = m_hi.net_energy_j(self.idle_w)
            tokens = self.n_gen * self.repeats
        else:
            lo_reps = max(1, self.repeats // 3)
            _, m_lo = self._run(self.command(gguf, 0, self.n_gen, lo_reps))
            net = m_hi.net_energy_j(self.idle_w) - m_lo.net_energy_j(self.idle_w)
            tokens = self.n_gen * (self.repeats - lo_reps)
            if tokens <= 0 or net <= 0:
                raise ProbeError("differential energy non-positive; increase repeats / run length")
        return SpeedEnergy(
            decode_tps=statistics.median(tg.samples_ts), prefill_tps=statistics.median(pp.samples_ts),
            joules_per_token=net / tokens, avg_power_w=m_hi.avg_power_w,
            decode_tps_spread=_mad_rel(tg.samples_ts), energy_spread=0.0, synthetic=(m_hi.backend in ("mock", "replay")),
        )
