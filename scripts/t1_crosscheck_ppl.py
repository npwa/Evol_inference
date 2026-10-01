"""T1 step 4: cross-backend perplexity check on the 3080 / CPU (Doc/implementation_plan_mac-m4.md §2.3).

Computes, for uniform FP16 / INT8 / INT4 genomes of Phi-3-mini:
  * HF + bitsandbytes (the base project's path), on BOTH windows: all 2047 scored tokens
    (what the README reports) and the second half only (what llama-perplexity scores);
  * llama.cpp via GgufAssembler + LlamaPerplexityProbe, CUDA build and CPU build.
and prints penalties relative to each path's own FP16.

    PYTHONPATH=. .venv/bin/python scripts/t1_crosscheck_ppl.py [--skip-hf] [--skip-cpu]
"""

import argparse
import json
import time
from pathlib import Path

from evol_inference.eval_data import write_wikitext2_test
from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.gguf_assembler import AssembledGgufProvider, GgufAssembler
from evol_inference.llama_probes import LlamaPerplexityProbe
from evol_inference.model_spec import get_spec

LLAMA = Path("~/work/llama.cpp").expanduser()
GGUF = Path("models/gguf")
UNIFORM = {p: [p] * N_SUPER_BLOCKS for p in Precision}


def hf_bnb() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evol_inference.fitness import load_fixed_eval_tokens
    from evol_inference.weight_bank import WeightBank

    d = "./models/phi-3-mini-4k-instruct"
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, device_map="cuda").eval()
    bank = WeightBank(model, device="cuda")
    ids = load_fixed_eval_tokens(tok, 2048, "cuda")
    out = {}
    for p, genome in UNIFORM.items():
        bank.assemble(genome)
        with torch.no_grad():
            lg = model(ids).logits.float()
        nll = -torch.log_softmax(lg[0, :-1], -1).gather(1, ids[0, 1:, None])[:, 0]
        out[p.value] = {"full_2047": nll.mean().exp().item(), "second_half_1023": nll[1024:].mean().exp().item()}
    del bank, model
    torch.cuda.empty_cache()
    return out


def gguf(build: str, ngl: int, work: Path, threads: int | None) -> dict:
    spec = get_spec("phi3-mini")
    sources = {Precision.FP16: GGUF / "phi3-mini-f16.gguf",
               Precision.INT8: GGUF / "phi3-mini-q8_0-pure.gguf",
               Precision.INT4: GGUF / "phi3-mini-q4_0-pure.gguf"}
    provider = AssembledGgufProvider(GgufAssembler(spec, sources), work)
    probe = LlamaPerplexityProbe(LLAMA / build / "bin/llama-perplexity", write_wikitext2_test(GGUF / "wikitext2_test.txt"),
                                 provider, n_gpu_layers=ngl, threads=threads)
    out = {}
    for p, genome in UNIFORM.items():
        t0 = time.perf_counter()
        out[p.value] = {"second_half_1023": probe.perplexity(genome), "wall_s": time.perf_counter() - t0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--out", default="results/t1_ppl_crosscheck.json")
    a = ap.parse_args()
    work = GGUF / "work_genome.gguf"
    res = {}
    # llama.cpp first: its CUDA processes need the 3080's memory, which the in-process
    # HF + bitsandbytes weight bank (all variants resident) does not give back reliably.
    res["llamacpp_cuda"] = gguf("build-cuda", 99, work, None)
    if not a.skip_cpu:
        res["llamacpp_cpu"] = gguf("build-cpu", 0, work, 8)
    if not a.skip_hf:
        res["hf_bnb"] = hf_bnb()
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for backend, rows in res.items():
        for window in ("full_2047", "second_half_1023"):
            if window not in rows["fp16"]:
                continue
            base = rows["fp16"][window]
            line = "  ".join(f"{p}={rows[p][window]:.4f} ({rows[p][window] / base - 1:+.2%})" for p in rows)
            print(f"{backend:14s} {window:17s} {line}")
        if "wall_s" in rows["fp16"]:
            print(f"{backend:14s} wall s/eval (assemble+load+ppl): " + ", ".join(f"{p}={rows[p]['wall_s']:.1f}" for p in rows))


if __name__ == "__main__":
    main()
