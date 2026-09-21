"""Phase 7: broader validation + Pareto plots (Doc/requirements.md §5/§8,
Doc/implementation_plan.md Phase 7).

1. Runs the broader WikiText-2 + C4 + Lambada perplexity validation pass on the
   baselines and the GA's final population (loaded from Phase 6's snapshot) -- cheap,
   a handful of configs, not every individual the search touched.
2. Generates two Pareto-front plots: the analytical byte-size metric the GA actually
   optimized against, and real measured wall-clock latency, to show the analytical
   proxy tracks something real (§8).

Plot color palette: 3-slot categorical (dataviz skill), the cap for scatter/all-pairs
chart forms -- GA population (blue), Pareto front (orange), baselines (aqua, always
individually text-labeled, since aqua's light-mode contrast needs that relief anyway).
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evol_inference.baselines import BASELINES
from evol_inference.pareto import pareto_front_indices
from evol_inference.validation import ValidationSuite
from evol_inference.weight_bank import Precision, WeightBank

MODEL_DIR = "./models/phi-3-mini-4k-instruct"
GA_SNAPSHOT = "./results/phase6_final.json"
VALIDATION_REPORT = "./results/phase7_validation.json"
PLOT_EFFICIENCY = "./results/pareto_efficiency.png"
PLOT_LATENCY = "./results/pareto_latency.png"

COLOR_GA = "#2a78d6"  # blue -- GA population
COLOR_PARETO = "#eb6834"  # orange -- Pareto front
COLOR_BASELINE = "#1baf7a"  # aqua -- baselines (always individually labeled)
TEXT_COLOR = "#1a1a19"


def load_snapshot() -> dict:
    return json.loads(open(GA_SNAPSHOT).read())


def genome_from_list(values: list[str]) -> list[Precision]:
    return [Precision(v) for v in values]


def make_plot(path, ga_x, ga_y, baseline_names, baseline_x, baseline_y, x_label, y_label, title, maximize_x):
    all_x = ga_x + baseline_x
    all_y = ga_y + baseline_y
    front_idx = set(pareto_front_indices(all_x, all_y, maximize_x=maximize_x, minimize_y=True))
    front_points = sorted((all_x[i], all_y[i]) for i in front_idx)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(ga_x, ga_y, s=32, color=COLOR_GA, alpha=0.55, label="GA population", zorder=2)

    if front_points:
        fx, fy = zip(*front_points)
        ax.plot(fx, fy, color=COLOR_PARETO, linewidth=1.75, zorder=3, label="Pareto front")
        ax.scatter(fx, fy, s=55, color=COLOR_PARETO, zorder=4, edgecolors="white", linewidths=0.75)

    ax.scatter(
        baseline_x, baseline_y, s=90, marker="D", color=COLOR_BASELINE, zorder=5,
        edgecolors="white", linewidths=0.75, label="Baselines",
    )
    # heuristic and uniform_int8 sit close together on the efficiency plot -- a
    # default uniform offset put their labels on top of each other (checked visually,
    # per the dataviz skill's final render-and-look pass), so nudge them apart.
    label_offsets = {"heuristic": (-8, -16), "uniform_int8": (10, 4)}
    for name, bx, by in zip(baseline_names, baseline_x, baseline_y):
        xytext = label_offsets.get(name, (6, 6))
        ha = "right" if xytext[0] < 0 else "left"
        ax.annotate(
            name, (bx, by), textcoords="offset points", xytext=xytext, fontsize=9,
            color=TEXT_COLOR, ha=ha,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    snapshot = load_snapshot()
    ga_individuals = [
        {"genome": genome_from_list(entry["genome"]), **entry["result"]}
        for entry in snapshot["population"]
    ]

    baseline_entries = []
    for name, genome in BASELINES.items():
        key = ",".join(p.value for p in genome)
        baseline_entries.append({"name": name, "genome": genome, **snapshot["cache"][key]})

    # --- Plots (pure data processing, no model needed -- Phase 6's snapshot already
    # has every fitness field) ---
    # Plot A is always the analytical view, computed from bytes_ directly: in
    # efficiency_metric="latency" mode the snapshot's efficiency_gain field is the
    # latency-based term, not the byte-size one this axis is labeled as.
    bytes_fp16 = next(b["bytes_"] for b in baseline_entries if b["name"] == "fp16")

    def analytical_gain(entry):
        return 1.0 - entry["bytes_"] / bytes_fp16

    print("Generating plots...")
    make_plot(
        PLOT_EFFICIENCY,
        ga_x=[analytical_gain(ind) for ind in ga_individuals],
        ga_y=[ind["accuracy_penalty"] for ind in ga_individuals],
        baseline_names=[b["name"] for b in baseline_entries],
        baseline_x=[analytical_gain(b) for b in baseline_entries],
        baseline_y=[b["accuracy_penalty"] for b in baseline_entries],
        x_label="efficiency_gain (analytical, 1 - bytes/bytes_fp16)",
        y_label="accuracy_penalty (perplexity_delta / baseline_perplexity)",
        title="Accuracy vs. efficiency -- analytical byte-size metric",
        maximize_x=True,
    )
    make_plot(
        PLOT_LATENCY,
        ga_x=[ind["latency_ms"] for ind in ga_individuals],
        ga_y=[ind["accuracy_penalty"] for ind in ga_individuals],
        baseline_names=[b["name"] for b in baseline_entries],
        baseline_x=[b["latency_ms"] for b in baseline_entries],
        baseline_y=[b["accuracy_penalty"] for b in baseline_entries],
        x_label="measured latency (ms/forward pass, GPU, report-only)",
        y_label="accuracy_penalty (perplexity_delta / baseline_perplexity)",
        title="Accuracy vs. efficiency -- real measured latency",
        maximize_x=False,
    )

    # --- Broader multi-dataset validation pass (needs the live model) ---
    print("\nLoading model for the broader validation pass...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16, device_map="cuda")
    model.eval()
    bank = WeightBank(model, device="cuda")
    suite = ValidationSuite(bank, tokenizer)

    report = {"baselines": {}, "ga_population": []}
    print("\nBaselines:")
    for b in baseline_entries:
        v = suite.evaluate(b["genome"])
        report["baselines"][b["name"]] = vars(v)
        print(
            f"  {b['name']:<14} wt2={v.wikitext2_perplexity:.4f}  c4={v.c4_perplexity:.4f}  "
            f"lambada={v.lambada_perplexity:.4f}  (fitness-loop wt2={b['perplexity']:.4f})"
        )

    print(f"\nValidating {len(ga_individuals)} GA population individuals...")
    for i, ind in enumerate(ga_individuals):
        v = suite.evaluate(ind["genome"])
        report["ga_population"].append({"genome": [p.value for p in ind["genome"]], **vars(v)})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ga_individuals)}")

    with open(VALIDATION_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()
