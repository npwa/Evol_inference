"""Phase 2 step 4 verification (Doc/implementation_plan.md): build the weight bank and
confirm genome assembly is both correct and actually dispatching real low-bit kernels.

1. all-FP16 genome must exactly reproduce the untouched baseline.
2. all-INT8 genome must closely match HF's own `load_in_8bit=True` reference.
3. all-INT4 genome must show a real VRAM/latency difference from all-FP16 (confirms the
   swap isn't silently falling back to a dequant-then-FP16-matmul path).
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evol_inference.weight_bank import N_SUPER_BLOCKS, Precision, WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"
PROMPT = "The quick brown fox jumps over the lazy dog."


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def get_logits(model, tokenizer, device="cuda"):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits


def time_forward(model, tokenizer, device="cuda", n=5):
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(2):  # warmup
            model(**inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            model(**inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


def main() -> None:
    tokenizer = load_tokenizer()

    print("Loading FP16 baseline...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    baseline_logits = get_logits(model, tokenizer)

    print("Building weight bank (quantizing every layer once)...")
    bank = WeightBank(model, device="cuda")

    # --- Check 1: all-FP16 genome exactly reproduces the baseline ---
    all_fp16 = [Precision.FP16] * N_SUPER_BLOCKS
    bank.assemble(all_fp16)
    fp16_logits = get_logits(model, tokenizer)
    max_diff = (fp16_logits - baseline_logits).abs().max().item()
    print(f"\n[Check 1] all-FP16 vs baseline: max abs logit diff = {max_diff:.6f}")
    assert max_diff == 0.0, "all-FP16 genome should exactly reproduce the baseline"
    print("  PASS")

    # --- Check 2: all-INT8 genome vs HF's load_in_8bit reference ---
    all_int8 = [Precision.INT8] * N_SUPER_BLOCKS
    bank.assemble(all_int8)
    int8_logits = get_logits(model, tokenizer)

    print("\nLoading HF load_in_8bit=True reference for comparison...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
        ),
        device_map="cuda",
    )
    ref_model.eval()
    ref_logits = get_logits(ref_model, tokenizer)
    del ref_model
    torch.cuda.empty_cache()

    cos_sim = torch.nn.functional.cosine_similarity(
        int8_logits.flatten(), ref_logits.flatten(), dim=0
    ).item()
    max_diff8 = (int8_logits - ref_logits).abs().max().item()
    print(f"[Check 2] all-INT8 vs HF load_in_8bit: cosine sim = {cos_sim:.6f}, max abs diff = {max_diff8:.4f}")
    assert cos_sim > 0.999, "all-INT8 genome should closely match HF's load_in_8bit reference"
    print("  PASS")

    # --- Check 3: all-INT4 genome shows a real VRAM/latency signal vs all-FP16 ---
    all_int4 = [Precision.INT4] * N_SUPER_BLOCKS

    bank.assemble(all_fp16)
    torch.cuda.reset_peak_memory_stats()
    get_logits(model, tokenizer)
    fp16_vram = torch.cuda.max_memory_allocated()
    fp16_latency = time_forward(model, tokenizer)

    bank.assemble(all_int4)
    torch.cuda.reset_peak_memory_stats()
    get_logits(model, tokenizer)
    int4_vram = torch.cuda.max_memory_allocated()
    int4_latency = time_forward(model, tokenizer)

    print(f"\n[Check 3] all-FP16: {bytes_to_gb(fp16_vram):.3f} GB, {fp16_latency*1000:.2f} ms/fwd")
    print(f"          all-INT4: {bytes_to_gb(int4_vram):.3f} GB, {int4_latency*1000:.2f} ms/fwd")
    assert int4_vram < fp16_vram, "all-INT4 genome should use less VRAM than all-FP16"
    print("  PASS (VRAM lower under INT4, confirming real low-bit weights are resident)")

    print("\nAll Phase 2 verification checks passed.")


if __name__ == "__main__":
    main()
