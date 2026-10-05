"""Aggregate results/pilot_s*.json into results/summary.md (tables for the paper)."""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    files = sorted(glob.glob(str(RES / "pilot_s*.json")))
    if not files:
        print("no results")
        return
    by_var = defaultdict(list)  # variant -> records (with k)
    llm_stats = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        llm_stats.append((d["scenario"], d.get("llm", {})))
        for r in d["records"]:
            for v, o in r["variants"].items():
                by_var[v].append((r["k"], o))
    ks = sorted({k for recs in by_var.values() for k, _ in recs})
    lines = [f"# Pilot summary ({len(files)} scenarios, {sum(len(v) for v in by_var.values())//max(1,len(by_var))} questions)\n"]
    lines.append("## Cell-level (retrieval) metrics\n")
    hdr = "| variant | mean |S| | ctx tokens | chain recall | answer-hop recall | " + " | ".join(f"chain@K={k}" for k in ks) + " |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for v, recs in by_var.items():
        row = [v, f"{mean([o['n_selected'] for _, o in recs]):.1f}", f"{mean([o['ctx_tokens'] for _, o in recs]):.0f}",
               f"{mean([float(o['chain_recall']) for _, o in recs]):.3f}", f"{mean([float(o['answer_hop_hit']) for _, o in recs if o['answer_hop_hit'] is not None]):.3f}"]
        for k in ks:
            row.append(f"{mean([float(o['chain_recall']) for kk, o in recs if kk == k]):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    # recall@k curves for one-shot rankers + per-step points for iterated variants
    curves = defaultdict(lambda: defaultdict(list))
    n_cells = []
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        n_cells.append(d["store"]["profiles"])
        for r in d["records"]:
            for m, rk in r.get("recall_at_k", {}).items():
                for k, v in rk.items():
                    curves[m][int(k)].append(float(v))
    if curves:
        kgrid = sorted(next(iter(curves.values())).keys())
        lines.append(f"\n## Full-chain recall vs budget k (one-shot rankers; store size mean {mean(n_cells):.0f} cells)\n")
        lines.append("| ranker | " + " | ".join(f"k={k}" for k in kgrid) + " |")
        lines.append("|" + "---|" * (len(kgrid) + 1))
        for m, c in curves.items():
            lines.append(f"| {m} | " + " | ".join(f"{mean(c[k]):.2f}" for k in kgrid) + " |")
        lines.append("\n## Iterated variants: mean |S| and chain recall per expansion step\n")
        lines.append("| variant | " + " | ".join(f"step {i}" for i in range(6)) + " |")
        lines.append("|" + "---|" * 7)
        for v, recs in by_var.items():
            if "trace_sizes" not in recs[0][1] or recs[0][1]["steps"] == 0 and not v.startswith(("prograph", "percand")):
                continue
            cells = []
            for i in range(6):
                sz = [o["trace_sizes"][min(i, len(o["trace_sizes"]) - 1)] for _, o in recs]
                rc = [float(o["trace_recall"][min(i, len(o["trace_recall"]) - 1)]) for _, o in recs]
                cells.append(f"{mean(sz):.0f} / {mean(rc):.2f}")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
    if any(o.get("judge") is not None for recs in by_var.values() for _, o in recs):
        lines.append("\n## Answer accuracy (LLM judge; n = judged questions, err = failed answer calls excluded)\n")
        hdr = "| variant | n | err | acc | " + " | ".join(f"K={k}" for k in ks) + " |"
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        for v, recs in by_var.items():
            js = [(k, o["judge"]) for k, o in recs if o.get("judge") is not None]
            if not js:
                continue
            n_err = sum(1 for _, o in recs if o.get("error"))
            row = [v, str(len(js)), str(n_err), f"{mean([j for _, j in js]):.3f}"] + [f"{mean([j for k, j in js if k == kk]):.3f}" for kk in ks]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("\n## LLM usage (write path + answers + judge, per run)\n")
    for sid, st in llm_stats:
        lines.append(f"- scenario {sid}: {st}")
    usage = RES / "usage.jsonl"
    if usage.exists():
        agg = defaultdict(lambda: [0, 0, 0])
        for line in open(usage, encoding="utf-8"):
            u = json.loads(line)
            a = agg[u.get("tag", "")]
            a[0] += 1; a[1] += u.get("prompt_tokens", 0); a[2] += u.get("completion_tokens", 0)
        lines.append("\n| tag | calls | prompt tokens | completion tokens |\n|---|---|---|---|")
        for t, (n, p, c) in sorted(agg.items()):
            lines.append(f"| {t} | {n} | {p} | {c} |")
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
