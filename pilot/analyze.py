"""Paper tables with paired-bootstrap confidence intervals over questions.

  python analyze.py            -> results/analysis.md
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
enc = tiktoken.get_encoding("cl100k_base")


def boot_ci(vals, n=2000, seed=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return m, (bs[int(0.025 * n)], bs[int(0.975 * n) - 1])


def paired_diff_ci(a, b, n=2000, seed=0):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan"), (float("nan"), float("nan")), 0
    rng = random.Random(seed)
    d = [x - y for x, y in pairs]
    m = sum(d) / len(d)
    bs = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return m, (bs[int(0.025 * n)], bs[int(0.975 * n) - 1]), len(pairs)


def main():
    files = sorted(glob.glob(str(RES / "pilot_s*.json")))
    recs = []
    conv_tokens = {}
    stores = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        sid = d["scenario"]
        stores[sid] = d["store"]
        conv = json.load(open(ROOT / "data" / "memhop" / "conversations" / f"scenario_{sid:02d}_sessions.json", encoding="utf-8"))
        conv_tokens[sid] = sum(len(enc.encode(t["text"])) for s in conv["sessions"] for t in s["turns"])
        for r in d["records"]:
            r["_sid"] = sid
            recs.append(r)
    variants = list(recs[0]["variants"].keys())
    L = [f"# Pilot analysis ({len(files)} scenarios, {len(recs)} questions)\n"]
    L.append("Store sizes (cells / residuals): " + ", ".join(f"s{s}: {v['profiles']}/{v['residuals']}" for s, v in stores.items()))
    L.append("Full-conversation tokens per scenario: " + ", ".join(f"s{s}: {t}" for s, t in conv_tokens.items()) + "\n")

    L.append("## Table A. Selection budget, cell-level chain recall, answer accuracy (95% bootstrap CIs)\n")
    L.append("| variant | mean cells | ctx tokens | ctx / full-conv | chain recall [CI] | answer acc [CI] (n) |")
    L.append("|---|---|---|---|---|---|")
    for v in variants:
        rows = [r["variants"][v] for r in recs if v in r["variants"]]
        cells = sum(o["n_selected"] for o in rows) / len(rows)
        ctx = sum(o["ctx_tokens"] for o in rows) / len(rows)
        ratio = sum(o["ctx_tokens"] / conv_tokens[r["_sid"]] for r, o in zip([r for r in recs if v in r["variants"]], rows)) / len(rows)
        cr, (cl, ch) = boot_ci([float(o["chain_recall"]) for o in rows])
        acc_vals = [o.get("judge") for o in rows]
        n_acc = sum(1 for x in acc_vals if x is not None)
        if n_acc:
            am, (al, ah) = boot_ci(acc_vals)
            acc_s = f"{am:.3f} [{al:.3f}, {ah:.3f}] ({n_acc})"
        else:
            acc_s = "—"
        L.append(f"| {v} | {cells:.1f} | {ctx:.0f} | {ratio:.1f}x | {cr:.3f} [{cl:.3f}, {ch:.3f}] | {acc_s} |")

    L.append("\n## Table B. Paired comparisons (one-shot vs iterated at matched budget; positive = first is better)\n")
    L.append("| comparison | metric | mean diff [95% CI] | n |")
    L.append("|---|---|---|---|")
    comps = [("crossenc_mpc", "percand"), ("cosine_mpc", "percand"), ("crossenc_k40", "percand_t0.3"),
             ("crossenc_k20", "cosine_h0"), ("crossenc_k20", "prograph"), ("cosine_mpc", "prograph")]
    for a_, b_ in comps:
        if a_ not in variants or b_ not in variants:
            continue
        for metric, key in [("chain recall", "chain_recall"), ("answer acc", "judge")]:
            A = [float(r["variants"][a_][key]) if r["variants"][a_].get(key) is not None else None for r in recs]
            B = [float(r["variants"][b_][key]) if r["variants"][b_].get(key) is not None else None for r in recs]
            m, (lo, hi), n = paired_diff_ci(A, B)
            if n:
                L.append(f"| {a_} − {b_} | {metric} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] | {n} |")

    L.append("\n## Table C. Expansion dynamics (mean cells after each step; chain recall)\n")
    L.append("| variant | step 0 | step 1 | step 2 | final | store |")
    L.append("|---|---|---|---|---|---|")
    for v in variants:
        rows = [r["variants"][v] for r in recs if v in r["variants"] and r["variants"][v].get("steps", 0) > 0 or v.startswith(("prograph", "percand"))]
        if not rows or "trace_sizes" not in rows[0]:
            continue
        def at(i):
            sz = sum(o["trace_sizes"][min(i, len(o["trace_sizes"]) - 1)] for o in rows) / len(rows)
            rc = sum(float(o["trace_recall"][min(i, len(o["trace_recall"]) - 1)]) for o in rows) / len(rows)
            return f"{sz:.0f} / {rc:.2f}"
        store_mean = sum(stores[r["_sid"]]["profiles"] for r in recs if v in r["variants"]) / len(rows)
        L.append(f"| {v} | {at(0)} | {at(1)} | {at(2)} | {at(99)} | {store_mean:.0f} |")

    L.append("\n## Cost ledger (from usage.jsonl)\n")
    agg = defaultdict(lambda: [0, 0, 0])
    for line in open(RES / "usage.jsonl", encoding="utf-8"):
        u = json.loads(line)
        a = agg[u["tag"]]; a[0] += 1; a[1] += u["prompt_tokens"]; a[2] += u["completion_tokens"]
    L.append("| tag | calls | prompt tokens | completion tokens |\n|---|---|---|---|")
    for t, (n, p, c) in sorted(agg.items()):
        L.append(f"| {t} | {n} | {p} | {c} |")
    (RES / "analysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
