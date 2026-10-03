"""Tier T3 report: turns results/t3_native_*.json + t3_bw_probe.jsonl + t3_qdot_bench.jsonl into the
tables and the roofline figure recorded in Doc/implementation_plan_mac-m4.md §22.

  * accuracy: KL divergence vs F16 (and the implied perplexity increase exp(KLD)-1) per KleidiAI configuration;
  * decode (tg128) achieved memory bandwidth vs the measured read-bandwidth roof (bw_probe), per precision/threads;
  * KleidiAI vs the fair baseline (the build without it, which keeps ggml's own Arm repack path);
  * qdot kernel throughput vs llama.cpp's effective single-thread bandwidth.

    PYTHONPATH=. python scripts/t3_report.py [--json results/t3_native_phi3-mini_<host>.json]
"""

import argparse
import glob
import json
import math
from pathlib import Path

# Bytes read per decoded token = every weight except the token-embedding table (only one row is touched).
# Measured from the Phi-3-mini GGUFs (models/gguf/phi3-mini-{f16,q8_0-pure,q4_0-pure}.gguf).
BYTES_PER_TOKEN = {"f16": 7.446e9, "q8": 3.956e9, "q4": 2.095e9}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--plot", default="results/t3_roofline.png")
    a = ap.parse_args()
    res = Path(a.results)
    path = Path(a.json) if a.json else Path(sorted(glob.glob(str(res / "t3_native_*.json")))[0])
    d = json.loads(path.read_text())
    if "platform" not in d:
        raise SystemExit("result has no platform block; refusing to report numbers with unknown provenance")
    bw = {r["threads"]: r["gb_per_s"] for r in load_jsonl(res / "t3_bw_probe.jsonl")}
    cfgs, thr = d["configs"], d["threads"]
    p = d["platform"]
    print(f"{path.name}: {p.get('host')} {p['machine']} kernel {p['release']}, {p['cpu_count']} CPUs, "
          f"{p.get('memory_gib')} GiB, features {[k for k, v in p['arm_features'].items() if v]}")
    print(f"memory read bandwidth (bw_probe): " + ", ".join(f"{t}t {g:.1f} GB/s" for t, g in sorted(bw.items())))

    print("\nACCURACY (KL divergence vs F16 logits; perplexity increase = exp(KLD)-1)")
    for c, x in cfgs.items():
        print(f"  {c:7s} Q8_0 {x['kld_q8']:.6f} ({100 * math.expm1(x['kld_q8']):5.2f}%)  Q4_0 {x['kld_q4']:.6f} "
              f"({100 * math.expm1(x['kld_q4']):5.2f}%)  kernels {x['kleidiai'] or '-'}")
    if "kai" in cfgs and "nokai" in cfgs:
        print(f"  KleidiAI Q8_0 is {cfgs['kai']['kld_q8'] / cfgs['nokai']['kld_q8']:.1f}x worse than the build without it")

    print("\nDECODE vs MEMORY ROOF (nokai = stock llama.cpp Arm path)")
    print(f"  {'prec':5s}{'thr':>4s}{'tok/s':>8s}{'GB/s':>8s}{'% of roof':>10s}")
    util = {}
    for prec in ("f16", "q8", "q4"):
        for t in thr:
            tps = cfgs["nokai"]["bench"][f"{prec}_t{t}"]["tg128_tps"]
            gbs = tps * BYTES_PER_TOKEN[prec] / 1e9
            util[(prec, t)] = (tps, gbs, 100 * gbs / bw[t])
            print(f"  {prec:5s}{t:>4d}{tps:8.2f}{gbs:8.1f}{util[(prec, t)][2]:9.0f}%")

    print("\nKLEIDIAI vs STOCK (ratio kai/nokai)")
    for prec in ("q8", "q4"):
        print("  " + prec + "  " + "   ".join(
            f"t={t}: decode {cfgs['kai']['bench'][f'{prec}_t{t}']['tg128_tps'] / cfgs['nokai']['bench'][f'{prec}_t{t}']['tg128_tps']:.2f}x "
            f"prefill {cfgs['kai']['bench'][f'{prec}_t{t}']['pp512_tps'] / cfgs['nokai']['bench'][f'{prec}_t{t}']['pp512_tps']:.2f}x" for t in thr))

    q = load_jsonl(res / "t3_qdot_bench.jsonl")
    one = {(prec): util[(prec, 1)][1] for prec in ("q8", "q4")} if 1 in thr else {}
    print("\nQDOT kernels (single thread, GB/s of weight bytes) vs llama.cpp effective single-thread decode bandwidth")
    for kern in ("q8_0_ref", "q8_0_sdot", "q8_0_smmla_2vec", "q4_0_ref", "q4_0_sdot"):
        v = [r["gb_per_s"] for r in q if r["kernel"] == kern]
        prec = "q8" if kern.startswith("q8") else "q4"
        ref = one.get(prec)
        print(f"  {kern:18s} {sum(v) / len(v):6.2f} GB/s" + (f"   = {100 * sum(v) / len(v) / ref:4.0f}% of llama.cpp's {ref:.1f} GB/s" if ref else ""))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        labels, colors = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_0"}, {"f16": "#6C7A89", "q8": "#0A7EA4", "q4": "#2E9E8A"}
        for prec in ("f16", "q8", "q4"):
            ax[0].plot(thr, [util[(prec, t)][1] for t in thr], "o-", color=colors[prec], label=labels[prec])
        ax[0].plot(sorted(bw), [bw[t] for t in sorted(bw)], "k--", label="read-bandwidth roof")
        ax[0].set(xlabel="threads", ylabel="effective weight bandwidth, GB/s", title="Phi-3 decode vs memory roof (c7g)")
        ax[0].legend(frameon=False)
        names = ["Q8_0 stock", "Q8_0 KleidiAI", "Q4_0 stock", "Q4_0 KleidiAI"]
        kld = [cfgs["nokai"]["kld_q8"], cfgs["kai"]["kld_q8"], cfgs["nokai"]["kld_q4"], cfgs["kai"]["kld_q4"]]
        ax[1].bar(names, [100 * math.expm1(k) for k in kld], color=["#0A7EA4", "#D1495B", "#2E9E8A", "#2E9E8A"])
        ax[1].set(ylabel="perplexity increase vs F16 (%)", title="Accuracy cost (KL divergence)")
        ax[1].tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"\nwrote {a.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
