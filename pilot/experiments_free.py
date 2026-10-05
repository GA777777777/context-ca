"""Three LLM-free follow-ups to the pilot (retrieval level only, local models).

  1. atomic   : does neighbourhood matter when cells do NOT pre-materialise neighbours?
                (1a) entity cells whose text is the residual bundle instead of the narrative
                (1b) atomic cells: every residual is its own cell (Mem0-style store)
  2. merged   : distractors — questions of scenario s run against the union store s1+s2+s3
  3. pairwise : budgeted expansion with a cross-encoder gate; separates
                (A) neighbourhood as candidate generator (score = ce(query, cand))
                (B) edge-conditioned evidence     (score = ce(query, seed-snippet || cand))
                against one-shot ce top-k at the same k.

  python experiments_free.py atomic|merged|pairwise|all [--scenarios 1,2,3] [--limit N]
Results: results/free_<exp>.json and results/free_summary.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from data import gold_chain, load_bundle, normalize
from metrics import hop_hits, n_tokens
from readpaths import ReadPaths, _unit
from run_pilot import RESULTS_DIR, STORE_DIR, make_embed_fn
from store import ProGraphStore

CE_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_GRID = [5, 10, 20, 40, 60, 85, 120, 200, 300, 441]


# ---------------------------------------------------------------- cross-encoder with disk cache
class CE:
    def __init__(self, path=RESULTS_DIR / "ce_cache.pkl"):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(CE_NAME, max_length=512)
        self.path = Path(path)
        self.cache: Dict[str, float] = pickle.load(open(self.path, "rb")) if self.path.exists() else {}
        self.n_new = 0

    @staticmethod
    def _k(q, t):
        return hashlib.md5((q + "\x1f" + t).encode("utf-8")).hexdigest()

    def scores(self, q: str, texts: List[str]) -> List[float]:
        keys = [self._k(q, t) for t in texts]
        todo = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in self.cache]
        if todo:
            sc = self.m.predict([(q, t) for _, t in todo], batch_size=16, show_progress_bar=False)
            for (i, _), s in zip(todo, sc):
                self.cache[keys[i]] = float(s)
            self.n_new += len(todo)
            if self.n_new % 2000 < len(todo):
                self.save()
        return [self.cache[k] for k in keys]

    def save(self):
        pickle.dump(self.cache, open(self.path, "wb"))


# ---------------------------------------------------------------- helpers
def matches(sel_names: List[str], hop_names: Set[str]) -> bool:
    return any(_name_match(c, n) for c in sel_names for n in hop_names)


def _name_match(cell_key: str, full_name: str) -> bool:
    k = normalize(cell_key); n = normalize(full_name); first = n.split()[0] if n.split() else n
    return k == n or k == first or (len(first) > 2 and re.fullmatch(re.escape(first) + r"( .*)?", k) is not None)


def chain_ok(sel_names: List[str], chain) -> bool:
    return all(matches(sel_names, h["names"]) if h["names"] else True for h in chain)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def load_store(sid, embed_fn):
    return ProGraphStore.load(STORE_DIR / f"scenario_{sid:02d}", None, embed_fn)


# ================================================================ EXP 1: atomic
def _load_ckpt(name, default):
    p = RESULTS_DIR / name
    if p.exists():
        try:
            d = json.load(open(p, encoding="utf-8"))
            return d
        except Exception:
            pass
    return default


def exp_atomic(scenarios, embed_fn, ce, limit):
    out = _load_ckpt("free_atomic.json", {"1a": [], "1b": []})
    if "1a" not in out:
        out = {"1a": [], "1b": []}
    done = {r["qa_id"] for r in out["1a"]}
    print(f"[atomic] resuming, {len(done)} questions already done", flush=True)
    for sid in scenarios:
        scenario, id2name, sessions, questions = load_bundle(sid)
        st = load_store(sid, embed_fn)
        # ---- 1a: entity cells, residual-bundle text (no narrative)
        view = ProGraphStore(None, embed_fn)
        view.display_names, view.types, view.known_names = st.display_names, st.types, st.known_names
        for k in st.profiles:
            res = st.residuals.get(k, [])
            if not res:
                continue
            view.profiles[k] = "\n".join(f"- {r}" for r in res)
            view.profile_emb[k] = _unit(np.mean(st.residual_emb[k], axis=0))
            view.residuals[k], view.residual_emb[k] = res, st.residual_emb[k]
        rp = ReadPaths(view, embed_fn)
        # ---- 1b: atomic residual cells
        atoms, atom_owner, atom_emb = [], [], []
        for k, res in st.residuals.items():
            for i, r in enumerate(res):
                atoms.append(r); atom_owner.append(k); atom_emb.append(st.residual_emb[k][i])
        atom_emb = np.vstack(atom_emb); atom_emb = atom_emb / (np.linalg.norm(atom_emb, axis=1, keepdims=True) + 1e-9)
        qs = questions[:limit] if limit else questions
        t0 = time.time()
        for qi, q in enumerate(qs):
            if q["id"] in done:
                continue
            chain = gold_chain(q, id2name)
            rec = {"qa_id": q["id"], "k": q["k"], "sid": sid, "variants": {}}
            base = rp.prograph(q["question"], M=5, tau=0.2, H=5)
            pc = rp.prograph(q["question"], M=5, tau=0.2, H=5, per_candidate=True)
            pc3 = rp.prograph(q["question"], M=5, tau=0.3, H=5, per_candidate=True)
            rk_cos = rp.ranking(q["question"], "cosine")
            keys = list(view.profiles)
            ce_sc = ce.scores(q["question"], [view.profiles[k][:2000] for k in keys])
            rk_ce = [k for k, _ in sorted(zip(keys, ce_sc), key=lambda x: -x[1])]
            b_pc = len(pc["selected"])
            variants = {"prograph": base["selected"], "percand": pc["selected"], "percand_t0.3": pc3["selected"],
                        "cosine_h0": rp.cosine_h0(q["question"], M=5)["selected"],
                        "cosine_k20": rk_cos[:20], "cosine_k40": rk_cos[:40], "cosine_mpc": rk_cos[:b_pc],
                        "crossenc_k20": rk_ce[:20], "crossenc_k40": rk_ce[:40], "crossenc_mpc": rk_ce[:b_pc]}
            for v, sel in variants.items():
                rec["variants"][v] = {"n": len(sel), "chain": chain_ok(sel, chain),
                                      "tokens": sum(n_tokens(view.profiles.get(k, "")) for k in sel)}
            rec["recall_at_k"] = {"cosine": {k: chain_ok(rk_cos[:k], chain) for k in K_GRID},
                                  "crossenc": {k: chain_ok(rk_ce[:k], chain) for k in K_GRID}}
            out["1a"].append(rec)
            # 1b atomic
            qv = _unit(embed_fn([q["question"]])[0])
            sc = atom_emb @ qv
            order = np.argsort(-sc)
            rec_b = {"qa_id": q["id"], "k": q["k"], "sid": sid, "n_atoms": len(atoms), "variants": {}, "recall_at_k": {}}
            # one-shot cosine top-k atoms; hop covered if an atom's owner matches the hop entity
            for kk in [10, 20, 40, 80, 160, 320]:
                sel_owner = [atom_owner[i] for i in order[:kk]]
                rec_b["variants"][f"atom_cos_k{kk}"] = {"n": kk, "chain": chain_ok(sel_owner, chain),
                                                        "tokens": sum(n_tokens(atoms[i]) for i in order[:kk])}
            # cross-encoder rerank of top-200 cosine atoms
            top = order[:200]
            ce_a = ce.scores(q["question"], [atoms[i] for i in top])
            re_order = [top[j] for j in np.argsort(-np.array(ce_a))]
            for kk in [10, 20, 40, 80]:
                sel_owner = [atom_owner[i] for i in re_order[:kk]]
                rec_b["variants"][f"atom_ce_k{kk}"] = {"n": kk, "chain": chain_ok(sel_owner, chain),
                                                       "tokens": sum(n_tokens(atoms[i]) for i in re_order[:kk])}
            # one-hop expansion over atoms: from top-k atoms, collect mentioned entities, add their top-r atoms by query cosine
            for kk, r_add in [(20, 4), (40, 4)]:
                sel = list(order[:kk]); owners = {atom_owner[i] for i in sel}
                mentioned = set()
                for i in sel:
                    for nm in st.find_names_in_text(atoms[i]):
                        mentioned.add(normalize(nm))
                for ent in mentioned - owners:
                    idx = [j for j, o in enumerate(atom_owner) if o == ent]
                    if not idx:
                        continue
                    idx = sorted(idx, key=lambda j: -sc[j])[:r_add]
                    sel += idx
                sel_owner = [atom_owner[i] for i in sel]
                rec_b["variants"][f"atom_cos_k{kk}_expand"] = {"n": len(sel), "chain": chain_ok(sel_owner, chain),
                                                               "tokens": sum(n_tokens(atoms[i]) for i in sel)}
            out["1b"].append(rec_b)
            if (qi + 1) % 20 == 0:
                print(f"[atomic s{sid}] {qi+1}/{len(qs)} ({time.time()-t0:.0f}s, ce new={ce.n_new})", flush=True)
                ce.save(); json.dump(out, open(RESULTS_DIR / "free_atomic.json", "w", encoding="utf-8"), indent=1)
        ce.save()
        json.dump(out, open(RESULTS_DIR / "free_atomic.json", "w", encoding="utf-8"), indent=1)  # checkpoint per scenario
    return out


# ================================================================ EXP 2: merged store with distractors
def exp_merged(scenarios, embed_fn, ce, limit):
    merged = ProGraphStore(None, embed_fn)
    per_sid = {}
    for sid in scenarios:
        st = load_store(sid, embed_fn)
        per_sid[sid] = st
        for k, p in st.profiles.items():
            mk = f"s{sid}::{k}"
            merged.profiles[mk] = p
            merged.profile_emb[mk] = st.profile_emb[k]
            merged.display_names[mk] = st.display_names.get(k, k)
            merged.types[mk] = st.types.get(k, "other")
            if k in st.residuals:
                merged.residuals[mk], merged.residual_emb[mk] = st.residuals[k], st.residual_emb[k]
        merged.known_names += [n for n in st.known_names if n not in merged.known_names]
    # name lookup must map to prefixed keys: patch find_names_in_text consumers by overriding normalize-based lookup
    name_to_keys = defaultdict(list)
    for mk in merged.profiles:
        name_to_keys[mk.split("::", 1)[1]].append(mk)

    class MergedRP(ReadPaths):
        def prograph(self, query, M=5, tau=0.2, H=5, per_candidate=False):
            selected, scores = self.stage1_m(query, M)
            trace = [sorted(selected)]
            qv = _unit(self.embed_fn([query])[0])
            for _ in range(H):
                cand, best, cs = set(), 0.0, {}
                for k in selected:
                    for name in self.s.find_names_in_text(self.s.profiles.get(k, "")):
                        for n in name_to_keys.get(normalize(name), []):
                            if n not in selected:
                                sc = float(np.dot(qv, _unit(self.s.profile_emb[n]))); cs[n] = sc; best = max(best, sc); cand.add(n)
                if not cand or best < tau:
                    break
                add = {n for n in cand if cs[n] >= tau} if per_candidate else cand
                if not add:
                    break
                selected |= add
                trace.append(sorted(selected))
            return {"selected": sorted(selected), "trace": trace, "steps": len(trace) - 1}

        def stage1_m(self, query, M=5, boost=0.3):
            scores = self.cosine_scores(query)
            for name in self.s.find_names_in_text(query):
                for n in name_to_keys.get(normalize(name), []):
                    if n in scores:
                        scores[n] += boost
            ranked = sorted(scores.items(), key=lambda x: -x[1])
            return {k for k, _ in ranked[:M]}, scores

    rp = MergedRP(merged, embed_fn)
    prev = _load_ckpt("free_merged.json", {"n_cells": 0, "records": []})
    out = prev.get("records", []) if prev.get("n_cells") == len(merged.profiles) else []
    done = {r["qa_id"] for r in out}
    print(f"[merged] resuming, {len(done)} questions already done", flush=True)
    keys = list(merged.profiles)
    for sid in scenarios:
        scenario, id2name, sessions, questions = load_bundle(sid)
        qs = questions[:limit] if limit else questions
        t0 = time.time()
        for qi, q in enumerate(qs):
            if q["id"] in done:
                continue
            chain = gold_chain(q, id2name)
            def own(sel):  # only cells from the question's own scenario can satisfy a hop
                return [k.split("::", 1)[1] for k in sel if k.startswith(f"s{sid}::")]
            def distract(sel):
                return sum(1 for k in sel if not k.startswith(f"s{sid}::")) / max(1, len(sel))
            base = rp.prograph(q["question"], tau=0.2)
            pc = rp.prograph(q["question"], tau=0.2, per_candidate=True)
            _, cos_scores = rp.stage1_m(q["question"], M=len(keys))
            rk_cos = [k for k, _ in sorted(cos_scores.items(), key=lambda x: -x[1])]
            ce_sc = ce.scores(q["question"], [merged.profiles[k][:2000] for k in keys])
            rk_ce = [k for k, _ in sorted(zip(keys, ce_sc), key=lambda x: -x[1])]
            b_pc = len(pc["selected"])
            variants = {"prograph": base["selected"], "percand": pc["selected"], "cosine_h0": sorted(rp.stage1_m(q["question"], 5)[0]),
                        "cosine_k20": rk_cos[:20], "cosine_k40": rk_cos[:40], "cosine_k85": rk_cos[:85], "cosine_mpc": rk_cos[:b_pc],
                        "crossenc_k20": rk_ce[:20], "crossenc_k40": rk_ce[:40], "crossenc_k85": rk_ce[:85], "crossenc_mpc": rk_ce[:b_pc]}
            rec = {"qa_id": q["id"], "k": q["k"], "sid": sid, "variants": {}}
            for v, sel in variants.items():
                rec["variants"][v] = {"n": len(sel), "chain": chain_ok(own(sel), chain), "distractor_frac": distract(sel),
                                      "tokens": sum(n_tokens(merged.profiles[k]) for k in sel)}
            rec["recall_at_k"] = {"cosine": {k: chain_ok(own(rk_cos[:k]), chain) for k in K_GRID},
                                  "crossenc": {k: chain_ok(own(rk_ce[:k]), chain) for k in K_GRID}}
            out.append(rec)
            if (qi + 1) % 20 == 0:
                print(f"[merged s{sid}] {qi+1}/{len(qs)} ({time.time()-t0:.0f}s, ce new={ce.n_new})", flush=True)
                ce.save(); json.dump({"n_cells": len(keys), "records": out}, open(RESULTS_DIR / "free_merged.json", "w", encoding="utf-8"), indent=1)
        ce.save()
        json.dump({"n_cells": len(keys), "records": out}, open(RESULTS_DIR / "free_merged.json", "w", encoding="utf-8"), indent=1)
    return out


# ================================================================ EXP 3: pairwise, budgeted expansion
def snippet(profile: str, name: str, width: int = 220) -> str:
    """Text window of the seed profile around the first mention of `name` (or its first token)."""
    pl = profile.lower(); nl = name.lower(); first = nl.split()[0] if nl.split() else nl
    i = pl.find(nl)
    if i < 0:
        m = re.search(r"\b" + re.escape(first) + r"\b", pl)
        i = m.start() if m else -1
    if i < 0:
        return profile[:width]
    return profile[max(0, i - width // 2): i + width // 2]


def exp_pairwise(scenarios, embed_fn, ce, limit, budgets=(10, 20, 40), per_step=5):
    out = _load_ckpt("free_pairwise.json", [])
    out = out if isinstance(out, list) and (not out or "variants" in out[0] and "oneshot_ce_k10" in out[0]["variants"]) else []
    done = {r["qa_id"] for r in out}
    print(f"[pairwise] resuming, {len(done)} questions already done", flush=True)
    for sid in scenarios:
        scenario, id2name, sessions, questions = load_bundle(sid)
        st = load_store(sid, embed_fn)
        rp = ReadPaths(st, embed_fn)
        keys = list(st.profiles)
        qs = questions[:limit] if limit else questions
        t0 = time.time()
        for qi, q in enumerate(qs):
            if q["id"] in done:
                continue
            chain = gold_chain(q, id2name)
            query = q["question"]
            seeds, _ = rp.stage1(query, M=5)
            qv = _unit(embed_fn([query])[0])
            ce_all = dict(zip(keys, ce.scores(query, [st.profiles[k][:2000] for k in keys])))
            rk_ce = [k for k, _ in sorted(ce_all.items(), key=lambda x: -x[1])]
            rec = {"qa_id": q["id"], "k": q["k"], "sid": sid, "variants": {}}

            def budgeted(score_fn, budget):
                sel = set(seeds); steps = 0; disp = st.display_names
                while len(sel) < budget:
                    frontier = {}  # cand -> (best score, seed)
                    for s_ in list(sel):
                        prof = st.profiles.get(s_, "")
                        for name in st.find_names_in_text(prof):
                            c = normalize(name)
                            if c in st.profiles and c not in sel:
                                sc = score_fn(s_, c, prof, name)
                                if c not in frontier or sc > frontier[c][0]:
                                    frontier[c] = (sc, s_)
                    if not frontier:
                        break
                    ranked = sorted(frontier.items(), key=lambda x: -x[1][0])
                    take = min(per_step, budget - len(sel))
                    for c, _ in ranked[:take]:
                        sel.add(c)
                    steps += 1
                    if steps > 12:
                        break
                return sorted(sel), steps

            def score_A(seed, cand, seed_prof, name):     # neighbourhood as candidate generator, one-cell CE score
                return ce_all[cand]

            def score_C(seed, cand, seed_prof, name):     # cosine control
                return float(np.dot(qv, _unit(st.profile_emb[cand])))

            pair_cache = {}
            def score_B(seed, cand, seed_prof, name):     # edge-conditioned: seed snippet || candidate profile
                key = (seed, cand)
                if key not in pair_cache:
                    txt = f"[{st.display_names.get(seed, seed)}] {snippet(seed_prof, name)} || [{st.display_names.get(cand, cand)}] {st.profiles[cand][:900]}"
                    pair_cache[key] = ce.scores(query, [txt])[0]
                return pair_cache[key]

            for b in budgets:
                rec["variants"][f"oneshot_ce_k{b}"] = {"n": b, "chain": chain_ok(rk_ce[:b], chain), "steps": 0}
                for tag, fn in [("nbr_ce", score_A), ("nbr_pair", score_B), ("nbr_cos", score_C)]:
                    sel, steps = budgeted(fn, b)
                    rec["variants"][f"{tag}_k{b}"] = {"n": len(sel), "chain": chain_ok(sel, chain), "steps": steps}
            out.append(rec)
            if (qi + 1) % 10 == 0:
                print(f"[pairwise s{sid}] {qi+1}/{len(qs)} ({time.time()-t0:.0f}s, ce new={ce.n_new})", flush=True)
                ce.save(); json.dump(out, open(RESULTS_DIR / "free_pairwise.json", "w", encoding="utf-8"), indent=1)
        ce.save()
        json.dump(out, open(RESULTS_DIR / "free_pairwise.json", "w", encoding="utf-8"), indent=1)
    return out


# ================================================================ summary
def summarize():
    L = ["# Free follow-up experiments (retrieval level, no LLM calls)\n"]
    p = RESULTS_DIR / "free_atomic.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8"))
        for tag, title in [("1a", "Exp 1a — entity cells with residual-bundle text (no narrative)"), ("1b", "Exp 1b — atomic residual cells (Mem0-style)")]:
            recs = d[tag]
            if not recs:
                continue
            L.append(f"## {title} (n = {len(recs)})\n")
            vs = list(recs[0]["variants"])
            L.append("| variant | mean cells | tokens | chain recall | K=2 | K=3 | K≥4 |\n|---|---|---|---|---|---|---|")
            for v in vs:
                rows = [(r["k"], r["variants"][v]) for r in recs if v in r["variants"]]
                L.append(f"| {v} | {mean([o['n'] for _, o in rows]):.1f} | {mean([o['tokens'] for _, o in rows]):.0f} | {mean([float(o['chain']) for _, o in rows]):.3f} | "
                         + " | ".join(f"{mean([float(o['chain']) for k, o in rows if cond(k)]):.2f}" for cond in (lambda k: k == 2, lambda k: k == 3, lambda k: k >= 4)) + " |")
            if recs[0].get("recall_at_k"):
                ks = list(recs[0]["recall_at_k"]["cosine"].keys())
                L.append("\n| ranker | " + " | ".join(f"k={k}" for k in ks) + " |\n|" + "---|" * (len(ks) + 1))
                for m in ("cosine", "crossenc"):
                    L.append(f"| {m} | " + " | ".join(f"{mean([float(r['recall_at_k'][m][k]) for r in recs]):.2f}" for k in ks) + " |")
            L.append("")
    p = RESULTS_DIR / "free_merged.json"
    if p.exists():
        d = json.load(open(p, encoding="utf-8")); recs = d["records"]
        L.append(f"## Exp 2 — merged store with distractors ({d['n_cells']} cells; n = {len(recs)})\n")
        L.append("| variant | mean cells | distractor frac | tokens | chain recall | K=2 | K=3 | K≥4 |\n|---|---|---|---|---|---|---|---|")
        for v in recs[0]["variants"]:
            rows = [(r["k"], r["variants"][v]) for r in recs]
            L.append(f"| {v} | {mean([o['n'] for _, o in rows]):.1f} | {mean([o['distractor_frac'] for _, o in rows]):.2f} | {mean([o['tokens'] for _, o in rows]):.0f} | {mean([float(o['chain']) for _, o in rows]):.3f} | "
                     + " | ".join(f"{mean([float(o['chain']) for k, o in rows if cond(k)]):.2f}" for cond in (lambda k: k == 2, lambda k: k == 3, lambda k: k >= 4)) + " |")
        ks = list(recs[0]["recall_at_k"]["cosine"].keys())
        L.append("\n| ranker | " + " | ".join(f"k={k}" for k in ks) + " |\n|" + "---|" * (len(ks) + 1))
        for m in ("cosine", "crossenc"):
            L.append(f"| {m} | " + " | ".join(f"{mean([float(r['recall_at_k'][m][k]) for r in recs]):.2f}" for k in ks) + " |")
        L.append("")
    p = RESULTS_DIR / "free_pairwise.json"
    if p.exists():
        recs = json.load(open(p, encoding="utf-8"))
        L.append(f"## Exp 3 — budgeted expansion with cross-encoder gates vs one-shot (n = {len(recs)})\n")
        L.append("| variant | mean cells | chain recall | K=1 | K=2 | K=3 | K≥4 |\n|---|---|---|---|---|---|---|")
        for v in recs[0]["variants"]:
            rows = [(r["k"], r["variants"][v]) for r in recs]
            L.append(f"| {v} | {mean([o['n'] for _, o in rows]):.1f} | {mean([float(o['chain']) for _, o in rows]):.3f} | "
                     + " | ".join(f"{mean([float(o['chain']) for k, o in rows if cond(k)]):.2f}" for cond in (lambda k: k == 1, lambda k: k == 2, lambda k: k == 3, lambda k: k >= 4)) + " |")
        L.append("")
    (RESULTS_DIR / "free_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["atomic", "merged", "pairwise", "all", "summary"])
    ap.add_argument("--scenarios", default="1,2,3")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.exp == "summary":
        summarize(); return
    scenarios = [int(x) for x in a.scenarios.split(",")]
    embed_fn = make_embed_fn()
    ce = CE()
    if a.exp in ("atomic", "all"):
        exp_atomic(scenarios, embed_fn, ce, a.limit)
    if a.exp in ("merged", "all"):
        exp_merged(scenarios, embed_fn, ce, a.limit)
    if a.exp in ("pairwise", "all"):
        exp_pairwise(scenarios, embed_fn, ce, a.limit)
    ce.save()
    if a.exp != "all":
        return  # keep per-exp processes small; run `summary` separately
    summarize()


if __name__ == "__main__":
    main()
