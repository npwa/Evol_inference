"""Offline analysis of a measured table (scripts/m4_measure_table.py output), at no hardware cost
(Doc/implementation_plan_mac-m4.md §22.6).

Per configuration: coverage of the genome space, the Pareto front over the measured genomes (accuracy, decode
speed, energy per token, all relative to ONE common baseline: the all-reference-precision genome under
--baseline-config), and, when the space is (nearly) fully measured, how well the GA and the sensitivity-greedy
rule recover that front. Across configurations: per-genome speed, energy and accuracy ratios (e.g. KleidiAI vs the
stock build). Results measured with a mock/replay meter are labelled SYNTHETIC.

    PYTHONPATH=. python scripts/m4_analyze.py --table results/m4_table_llama3.1-8b.jsonl --model llama3.1-8b
"""

import argparse
import json
import statistics
from pathlib import Path

from evol_inference.genome import N_SUPER_BLOCKS, Precision
from evol_inference.mac_measure import greedy_chain
from evol_inference.measured_table import MeasuredTable
from evol_inference.mo_fitness import MultiObjectiveEvaluator
from evol_inference.mo_ga import MOSteadyStateGA, ParetoRanker, run_mo_search
from evol_inference.model_spec import get_spec
from evol_inference.objectives import DEFAULT_HV_REF
from evol_inference.sensitivity import SensitivityTable
from evol_inference.tabulated import enumerate_genomes, front_quality, genome_code, parse_code, true_front


_WARNED: set[str] = set()


def reference_for(table: MeasuredTable, spec):
    """The model's reference genome if it was measured, else the uniform genome of the highest-bit precision in the
    alphabet that was (with a warning: gains are then relative to that genome, not to the F16 reference)."""
    ref = spec.reference_genome()
    if table.has(ref):
        return ref
    for p in sorted(spec.alphabet, key=lambda x: -x.bits):
        g = [p] * N_SUPER_BLOCKS
        if table.has(g):
            if table.config not in _WARNED:
                _WARNED.add(table.config)
                print(f"WARNING [{table.config}]: reference genome {genome_code(ref)} was not measured; "
                      f"baseline is {genome_code(g)} instead (gains/penalties are relative to it)")
            return g
    raise SystemExit(f"[{table.config}] neither the reference genome nor any uniform genome was measured: no baseline")


def evaluator(table: MeasuredTable, spec, baseline_table: MultiObjectiveEvaluator | None):
    ev = MultiObjectiveEvaluator(table, table, reference_for(table, spec), baseline_repeats=1)
    if baseline_table is not None:
        ev.baseline = baseline_table.baseline  # one common baseline across configurations
    return ev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--baseline-config", default=None, help="configuration whose reference-genome measurement is the common baseline (default: stock if present)")
    ap.add_argument("--alphabet", nargs="*", choices=[p.value for p in Precision], default=None,
                    help="restrict the analysed space (and the GA) to these precisions, e.g. int8 int4 for a table measured with m4_measure_table.py --alphabet int8 int4; "
                         "the reference genome stays the baseline. Default: the model's alphabet")
    ap.add_argument("--sensitivity", default=None)
    ap.add_argument("--studies", type=int, default=10)
    ap.add_argument("--min-coverage", type=float, default=0.9, help="run GA/greedy studies only above this fraction of the space")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = get_spec(a.model)
    alphabet = [Precision(x) for x in a.alphabet] if a.alphabet else list(spec.alphabet)
    configs = a.configs or MeasuredTable.configs_in(a.table)
    tables = {c: MeasuredTable.from_jsonl(a.table, c) for c in configs}
    base_cfg = a.baseline_config or ("stock" if "stock" in tables else configs[0])
    base_ev = evaluator(tables[base_cfg], spec, None)
    space = len(list(enumerate_genomes(alphabet)))
    synthetic = any(t.synthetic for t in tables.values())
    report: dict = {"model": spec.key, "baseline_config": base_cfg, "synthetic": synthetic, "space": space, "alphabet": [x.value for x in alphabet], "configs": {}}
    if synthetic:
        print("*** SYNTHETIC: rows measured with a mock/replay energy meter (energy_gain is not a measurement) ***")
    print(f"{spec.key}: common baseline = {base_cfg} reference genome "
          f"(decode {base_ev.baseline.decode_tps:.2f} tok/s, {base_ev.baseline.joules_per_token:.4f} J/tok)")

    sens = Path(a.sensitivity) if a.sensitivity else Path(f"results/sensitivity_{spec.key}_kld.json")
    chain = greedy_chain(spec, SensitivityTable.from_json(sens)) if sens.exists() else []

    for c, t in tables.items():
        ev = evaluator(t, spec, base_ev)
        pts = {code: ev.evaluate(parse_code(code)).minimized() for code in t.codes() if set(parse_code(code)) <= set(alphabet)}
        front = true_front(pts)
        cov = len(pts) / space
        hv = front_quality(list(front.values()), list(front.values()), DEFAULT_HV_REF)["hv_true"]
        entry = {"measured": len(pts), "coverage": cov, "front_size": len(front), "hypervolume": hv, "front": []}
        print(f"\n[{c}] {len(pts)}/{space} genomes ({100 * cov:.0f}%), Pareto front {len(front)} points, HV {hv:.4f}")
        print("  genome    dAcc%    decode   energy")
        for code, p in sorted(front.items(), key=lambda kv: kv[1][0]):
            v = ev.evaluate(parse_code(code))
            entry["front"].append({"genome": code, "acc_penalty": v.accuracy_penalty, "speed_gain": v.speed_gain, "energy_gain": v.energy_gain})
            print(f"  {code}  {100 * v.accuracy_penalty:7.3f}  {v.speed_gain:+7.3f}  {v.energy_gain:+7.3f}")
        if cov >= a.min_coverage:
            if chain:
                g = [x for x in chain if t.has(x) and set(x) <= set(alphabet)]
                q = front_quality([pts[genome_code(x)] for x in g], list(front.values()), DEFAULT_HV_REF)
                entry["greedy"] = q
                print(f"  sensitivity-greedy chain ({len(g)} genomes): HV ratio {q['hv_ratio']:.4f}, recall {q['recall']:.2f}")
            ratios = []
            if len(pts) < space:
                print(f"  (GA study needs every genome of the space measured: {space - len(pts)} missing under {c!r}; skipped)")
            for seed in range(a.studies if len(pts) == space else 0):
                e = evaluator(t, spec, base_ev)
                ga = MOSteadyStateGA(e, capacity=40, ranker=ParetoRanker(), seed=seed, alphabet=alphabet)
                ga.seed([x for x in spec.baselines().values() if t.has(x) and set(x) <= set(alphabet)])
                run_mo_search(ga, 400, stagnation_limit=10_000)
                found = [e.evaluate(m.genome).minimized() for m in ga.population.front()]
                ratios.append(front_quality(found, list(front.values()), DEFAULT_HV_REF)["hv_ratio"])
            if ratios:
                entry["ga_hv_ratio"] = {"mean": statistics.mean(ratios), "min": min(ratios), "max": max(ratios)}
                print(f"  GA (400 evals, {a.studies} seeds) HV ratio vs front: {statistics.mean(ratios):.4f} [{min(ratios):.4f}..{max(ratios):.4f}]")
        else:
            print(f"  (coverage below {100 * a.min_coverage:.0f}%: GA / greedy studies skipped)")
        report["configs"][c] = entry

    if len(tables) > 1:
        print(f"\nCross-configuration ratios on common genomes (configuration / {base_cfg}):")
        names = [base_cfg] + [c for c in tables if c != base_cfg]
        for other in names[1:]:
            common = [parse_code(x) for x in tables[names[0]].codes() if tables[other].has(parse_code(x))]
            if not common:
                continue
            sp = [tables[other].measure(g).decode_tps / tables[names[0]].measure(g).decode_tps for g in common]
            pf = [tables[other].measure(g).prefill_tps / tables[names[0]].measure(g).prefill_tps for g in common]
            en = [tables[other].measure(g).joules_per_token / tables[names[0]].measure(g).joules_per_token for g in common]
            kl = [(tables[other].kld(g) + 1e-9) / (tables[names[0]].kld(g) + 1e-9) for g in common]
            row = {"n": len(common), "decode_median": statistics.median(sp), "prefill_median": statistics.median(pf),
                   "energy_median": statistics.median(en), "kld_ratio_median": statistics.median(kl), "kld_ratio_max": max(kl)}
            report.setdefault("cross", {})[f"{other}/{names[0]}"] = row
            print(f"  {other}/{names[0]} (n={len(common)}): decode x{row['decode_median']:.2f}, prefill x{row['prefill_median']:.2f}, "
                  f"J/token x{row['energy_median']:.2f}, KLD median x{row['kld_ratio_median']:.2f} (max x{row['kld_ratio_max']:.1f})")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
