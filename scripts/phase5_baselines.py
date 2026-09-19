"""Phase 5: evaluate and report the baseline configurations (Doc/requirements.md §7).

These are what the GA (Phase 6) needs to beat -- specifically the heuristic baseline,
per §7 ("this is what the GA needs to beat to make the project's point").
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.baselines import BASELINES
from evol_inference.fitness import FitnessEvaluator
from evol_inference.weight_bank import WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    model.eval()

    bank = WeightBank(model, device="cuda")
    evaluator = FitnessEvaluator(bank, tokenizer, measure_latency=True)

    header = f"{'baseline':<14} {'perplexity':>11} {'acc_penalty':>12} {'eff_gain':>9} {'fitness':>8} {'GB':>7} {'ms/fwd':>8}"
    print(header)
    print("-" * len(header))
    for name, genome in BASELINES.items():
        r = evaluator.evaluate(genome)
        gb = r.bytes_ / (1024**3)
        print(
            f"{name:<14} {r.perplexity:>11.4f} {r.accuracy_penalty:>12.4f} "
            f"{r.efficiency_gain:>9.4f} {r.fitness:>8.4f} {gb:>7.3f} {r.latency_ms:>8.2f}"
        )


if __name__ == "__main__":
    main()
