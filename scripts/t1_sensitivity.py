"""T1 step 5: per-block sensitivity table (Doc/implementation_plan_mac-m4.md §2.2.2).

llama.cpp path: uniform GGUFs (`llama-quantize --pure`) -> GgufAssembler -> llama-perplexity
on the 3080. Optional `--hf-bnb` adds the bitsandbytes path on the *same* scored window (second
half of the first 2048-token chunk) for a like-for-like profile comparison (Phi-3 only).

    PYTHONPATH=. .venv/bin/python scripts/t1_sensitivity.py --model phi3-mini [--chunks 2] [--hf-bnb]
"""

import argparse
import time
from pathlib import Path

from evol_inference.eval_data import scored_tokens, write_wikitext2_test
from evol_inference.genome import Precision
from evol_inference.gguf_assembler import AssembledGgufProvider, GgufAssembler
from evol_inference.llama_probes import LlamaKldProbe, LlamaPerplexityProbe
from evol_inference.model_spec import get_spec
from evol_inference.sensitivity import SensitivityTable, edge_dominance, measure_sensitivity, spearman

LLAMA = Path("~/work/llama.cpp").expanduser()
GGUF = Path("models/gguf")


def llamacpp_table(spec, build: str, ngl: int, chunks: int, work_dir: Path, metric: str = "ppl") -> SensitivityTable:
    names = {Precision.FP16: "f16", Precision.INT8: "q8_0-pure", Precision.INT4: "q4_0-pure"}
    sources = {p: GGUF / f"{spec.key}-{n}.gguf" for p, n in names.items() if (GGUF / f"{spec.key}-{n}.gguf").exists()}
    prov = AssembledGgufProvider(GgufAssembler(spec, sources), work_dir / f"work_{spec.key}.gguf")
    ref = spec.reference_precision
    text = write_wikitext2_test(GGUF / "wikitext2_test.txt")
    exe = LLAMA / build / "bin/llama-perplexity"
    if metric == "kld":
        # base logits are specific to (model, reference, text, ctx, chunks, build): encode in the name
        base = work_dir / f"base_{spec.key}_{ref.value}_c2048_n{chunks}_{build}.kld"
        base_model = sources[ref]  # the original model: F16 GGUF (Q8_0 for models whose F16 does not fit)
        probe = LlamaKldProbe(exe, text, prov, base, base_model, chunks=chunks, n_gpu_layers=ngl)
    else:
        probe = LlamaPerplexityProbe(exe, text, prov, chunks=chunks, n_gpu_layers=ngl)
    precs = [p for p in (Precision.INT8, Precision.INT4) if p is not ref]
    t0 = time.perf_counter()
    t = measure_sensitivity(probe.perplexity, ref, precs, meta={
        "model": spec.key, "backend": f"llama.cpp {build}", "ctx": 2048, "chunks": chunks,
        "scored_tokens": scored_tokens(2048, chunks)})
    t.meta["wall_s"] = round(time.perf_counter() - t0, 1)
    t.meta["metric"] = metric
    if metric == "kld":
        # Noise floor: the base file against its own saved logits (should be ~0), the KLD of the
        # reference genome (cost of the quantized embeddings/head), and per-genome standard errors.
        t.meta["kld_floor"] = probe.measure_file(base_model).mean_kld  # base file vs its own saved logits
        t.meta["kld_reference_genome"] = probe.measure(spec.reference_genome()).mean_kld  # fixed-tensor cost
        t.meta["kld_stderr"] = {k: v.kld_stderr for k, v in
                                ((",".join(g), r) for g, r in probe.history.items())}
        t.meta["same_top_p"] = {",".join(g): r.same_top_p for g, r in probe.history.items()}
    return t


def bnb_table(chunks: int, metric: str = "ppl") -> SensitivityTable:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evol_inference.fitness import load_fixed_eval_tokens
    from evol_inference.weight_bank import WeightBank

    d = "./models/phi-3-mini-4k-instruct"
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, device_map="cuda").eval()
    bank = WeightBank(model, device="cuda")
    ids = load_fixed_eval_tokens(tok, 2048, "cuda")

    def second_half_logp(genome):
        bank.assemble(genome)
        with torch.no_grad():
            lg = model(ids).logits.float()
        return torch.log_softmax(lg[0, :-1], -1)[1024:], ids[0, 1:][1024:]

    # base = the all-FP16 genome (here that IS the original model: bitsandbytes leaves the
    # embeddings / head / norms in FP16, unlike the GGUF genomes' quantized fixed tensors)
    base_logp, tgt = second_half_logp([Precision.FP16] * 8)
    ppl0 = (-base_logp.gather(1, tgt[:, None])[:, 0]).mean().exp().item()

    def metric_fn(genome) -> float:
        logp, _ = second_half_logp(genome)
        if metric == "kld":  # mean_t KL(p_base || p_q) over the scored positions, as llama-perplexity computes it
            return ppl0 * torch.exp((base_logp.exp() * (base_logp - logp)).sum(-1).mean()).item()
        return (-logp.gather(1, tgt[:, None])[:, 0]).mean().exp().item()

    return measure_sensitivity(metric_fn, Precision.FP16, meta={
        "model": "phi3-mini", "backend": "HF+bitsandbytes", "metric": metric,
        "window": "second half of 2048 (1023 tokens)"})


def show(t: SensitivityTable, label: str) -> None:
    print(f"\n== {label}  ({t.meta.get('backend')}, ref ppl {t.reference_perplexity:.4f}, {t.meta.get('scored_tokens', '')} tokens)")
    for p, row in t.penalty.items():
        print(f"  {p:5s} penalty %: " + " ".join(f"{100 * x:6.3f}" for x in row) +
              f"  | sum {100 * sum(row):.3f}  uniform {100 * t.uniform_penalty[p]:.3f}  "
              f"residual {100 * t.additivity_residual(Precision(p)):+.3f}  edge/interior {edge_dominance(row):.1f}x")
    for p in t.penalty:
        print(f"  {p:5s} block share % : " + " ".join(f"{100 * x:6.1f}" for x in t.share(Precision(p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phi3-mini")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--build", default="build-cuda")
    ap.add_argument("--hf-bnb", action="store_true")
    ap.add_argument("--metric", choices=["ppl", "kld"], default="ppl",
                    help="ppl: perplexity delta (noisy); kld: KL divergence to the reference logits (plan §16)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--work-dir", default=None, help="where the reused assembled GGUF is written; default "
                    "/dev/shm if present (RAM-backed: avoids disk write-back throttling, see plan §16)")
    a = ap.parse_args()
    spec = get_spec(a.model)
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)

    work = Path(a.work_dir) if a.work_dir else (Path("/dev/shm") if Path("/dev/shm").is_dir() else GGUF)
    t = llamacpp_table(spec, a.build, 99 if "cuda" in a.build else 0, a.chunks, work, a.metric)
    suffix = ("" if a.chunks == 1 else f"_chunks{a.chunks}") + ("" if a.build == "build-cuda" else f"_{a.build.removeprefix('build-')}") + ("" if a.metric == "ppl" else "_kld")
    t.to_json(out / f"sensitivity_{spec.key}{suffix}.json")
    show(t, f"{spec.key} llama.cpp")
    if a.hf_bnb:
        b = bnb_table(a.chunks, a.metric)
        b.to_json(out / f"sensitivity_{spec.key}_bnb{'' if a.metric == 'ppl' else '_kld'}.json")
        show(b, f"{spec.key} bnb (same window)")
        for p in ("int8", "int4"):
            print(f"  Spearman(llama.cpp vs bnb, {p}) = {spearman(t.penalty[p], b.penalty[p]):.3f}")


if __name__ == "__main__":
    main()
