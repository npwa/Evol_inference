"""Phase 1: load the FP16 baseline, check VRAM usage, and enumerate the quantizable
linear layers per transformer block, grouped into the 8 super-blocks from
Doc/requirements.md §4. Doc/implementation_plan.md Phase 1, steps 2-3.
"""

import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./models/phi-3-mini-4k-instruct"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
N_SUPER_BLOCKS = 8


def bytes_to_gb(n: int) -> float:
    return n / (1024**3)


def main() -> None:
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    after_load = torch.cuda.memory_allocated()
    print(f"VRAM after load: {bytes_to_gb(after_load):.2f} GB (expected ~7.6GB, §3)")

    # --- Step 2: one forward pass, confirm VRAM stays in budget ---
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)

    peak = torch.cuda.max_memory_allocated()
    print(f"Peak VRAM after one forward pass: {bytes_to_gb(peak):.2f} GB")
    print(f"Usable budget (§3): ~9.5GB")

    # --- Step 3: enumerate quantizable linear layers per transformer block ---
    per_block_linears: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            match = LAYER_RE.match(name)
            if match:
                block_idx = int(match.group(1))
                n_params = sum(p.numel() for p in module.parameters())
                per_block_linears[block_idx].append((name, n_params))

    n_blocks = len(per_block_linears)
    print(f"\nTransformer blocks found: {n_blocks} (§3/§4 assume 32)")

    # Compare structure (suffix after "model.layers.<idx>.") not full names, so
    # e.g. block 0's "self_attn.qkv_proj" and block 5's are treated as the same shape.
    suffix_re = re.compile(r"^model\.layers\.\d+\.(.+)$")
    structure_per_block = {
        idx: tuple(sorted(suffix_re.match(n).group(1) for n, _ in layers))
        for idx, layers in per_block_linears.items()
    }
    distinct_shapes = set(structure_per_block.values())
    print(f"Distinct per-block linear-layer structures: {len(distinct_shapes)} (expect 1 if uniform)")
    for shape in distinct_shapes:
        print(f"  {shape}")

    if n_blocks % N_SUPER_BLOCKS != 0:
        print(
            f"\nWARNING: {n_blocks} blocks does not divide evenly into "
            f"{N_SUPER_BLOCKS} super-blocks (§4 assumed it would)."
        )
    else:
        blocks_per_super = n_blocks // N_SUPER_BLOCKS
        print(f"\n8 super-blocks of {blocks_per_super} transformer blocks each:")
        for sb in range(N_SUPER_BLOCKS):
            block_ids = range(sb * blocks_per_super, (sb + 1) * blocks_per_super)
            sb_params = sum(
                p for idx in block_ids for _, p in per_block_linears[idx]
            )
            print(
                f"  super-block {sb}: transformer blocks {list(block_ids)}, "
                f"{sb_params:,} quantizable params "
                f"({bytes_to_gb(sb_params * 2):.3f} GB at FP16)"
            )


if __name__ == "__main__":
    main()
