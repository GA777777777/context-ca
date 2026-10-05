"""Write-path propagation experiment: does a state change reach every cell that holds the old fact?

For each scenario, inject a session 21 in which person A tells a third person C that A and B
are no longer <relation> and that B moved away. Then compare write rules by how many cells
that asserted the old relation get updated, at what rewrite cost.

Rules
  R0 prograph : rewrite only entities identified in / mentioned by the injected session (as released)
  R1 comention: R0 + every cell whose text mentions both A and B (structural 1-hop)
  R2 ca_gate  : excitation spreads from A,B over co-mention neighbours (<= 2 hops); a cross-encoder gate
                (change sentence vs cell text) decides which excited cells are rewritten
  R3 oracle   : rewrite exactly the cells an LLM judge says asserted the old relation (upper bound)

Metrics (LLM judge on profile text and on residuals, per cell)
  affected      = cells asserting the old relation before injection (judge on pre-state)
  stale_profile = affected cells whose profile still asserts the old relation after the rule ran
  new_fact      = affected cells whose profile reflects the change
  stale_resid   = affected cells with a residual still asserting the old relation (never evicted)
  rewrites / tokens per rule

  python write_propagation.py --scenarios 1,2,3 --per-scenario 4 [--limit-cells 40]
Results: results/prop_s<sid>.json, results/prop_summary.md
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from data import load_bundle, normalize
from llm import LLM
from metrics import n_tokens
from run_pilot import RESULTS_DIR, STORE_DIR, make_embed_fn
from store import ProGraphStore, _PROFILE_UPDATE_SYSTEM, _PROFILE_UPDATE_USER, _parse_profile_and_residuals

CITIES = ["Denver", "Austin", "Seattle", "Chicago", "Portland", "Atlanta"]
JUDGE_SYS = "You are a precise reader of a knowledge profile. Answer with JSON only, no prose."
JUDGE_USER = """Text about an entity:
{text}

Facts to check:
(old) {A} and {B} are currently {relation}.
(new) {A} and {B} are no longer {relation}, and/or {B} moved to {city}.

Does the text state or clearly imply each fact? Answer JSON exactly: {{"old": true|false, "new": true|false}}"""


def humanize(rel: str) -> str:
    return rel.replace("_", " ")


def make_injection(scenario, id2name, idx, sid):
    rels = [r for r in scenario["relationships"] if not re.search(r"sibling|parent|mother|father|cousin|family", r["relation"])]
    r = rels[idx % len(rels)]
    A, B = id2name[r["person_a"]], id2name[r["person_b"]]
    others = [p["name"] for p in scenario["people"] if p["name"] not in (A, B)]
    C = others[idx % len(others)]
    city = CITIES[(sid * 7 + idx) % len(CITIES)]
    rel = humanize(r["relation"])
    a1, c1 = A.split()[0], C.split()[0]
    turns = [
        {"speaker": A, "text": f"Hey {c1}, some news. {B} and I are no longer {rel}. It ended two weeks ago."},
        {"speaker": C, "text": f"Really? What happened, {a1}?"},
        {"speaker": A, "text": f"{B} moved to {city} for a new job last month. We are still on good terms, but we are not {rel} anymore."},
        {"speaker": C, "text": f"That is a big change. I hope {B.split()[0]} likes {city}."},
        {"speaker": A, "text": f"I think so. Anyway, from now on if anyone asks, {B} and I are not {rel}."},
    ]
    for t in turns:
        t["timestamp"] = "2024-09-15"
    return {"A": A, "B": B, "C": C, "relation": rel, "city": city, "turns": turns}


def clone_store(base: ProGraphStore, llm, embed_fn) -> ProGraphStore:
    """Data-only copy (the store holds an LLM client with a lock, so deepcopy fails)."""
    s = ProGraphStore(llm, embed_fn)
    s.profiles = dict(base.profiles)
    s.display_names, s.types = dict(base.display_names), dict(base.types)
    s.profile_emb = {k: v.copy() for k, v in base.profile_emb.items()}
    s.residuals = {k: list(v) for k, v in base.residuals.items()}
    s.residual_emb = {k: v.copy() for k, v in base.residual_emb.items()}
    s.known_names, s.session_count = list(base.known_names), base.session_count
    return s


def cell_text(st: ProGraphStore, k: str) -> str:
    return st.profiles.get(k, "") + ("\n" + "\n".join(st.residuals.get(k, [])) if st.residuals.get(k) else "")


def mentions(text: str, name: str) -> bool:
    tl, nl = text.lower(), name.lower()
    first = nl.split()[0] if nl.split() else nl
    return nl in tl or (len(first) > 2 and re.search(r"\b" + re.escape(first) + r"\b", tl) is not None)


def judge_cell(llm, text, inj) -> dict:
    resp = llm(JUDGE_SYS, JUDGE_USER.format(text=text[:6000], A=inj["A"], B=inj["B"], relation=inj["relation"], city=inj["city"]), tag="prop:judge")
    m = re.search(r"\{.*?\}", resp, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return {"old": bool(d.get("old", False)), "new": bool(d.get("new", False))}
    except Exception:
        return {"old": "true" in resp.lower().split("old")[-1][:20], "new": False}


EVICT_SYS = "You are a precise reader. Answer with JSON only."
EVICT_USER = """Numbered facts about {name}:
{facts}

Which facts state or clearly imply that {A} and {B} are currently {relation} (a fact that is now outdated)?
Answer JSON exactly: {{"stale": [list of fact numbers]}}"""


def evict_residuals(llm, st: ProGraphStore, keys, inj, workers=6):
    """Contradiction as a local transition: drop residuals of the given cells that assert the old fact."""
    def _one(k):
        res = st.residuals.get(k, [])
        if not res:
            return k, []
        facts = "\n".join(f"{i+1}. {r}" for i, r in enumerate(res))
        resp = llm(EVICT_SYS, EVICT_USER.format(name=st.display_names.get(k, k), facts=facts[:8000], A=inj["A"], B=inj["B"], relation=inj["relation"]), tag="prop:evict")
        m = re.search(r"\{.*?\}", resp, re.S)
        try:
            idx = [int(i) - 1 for i in json.loads(m.group(0)).get("stale", [])] if m else []
        except Exception:
            idx = []
        return k, [i for i in idx if 0 <= i < len(res)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, keys))
    n = 0
    for k, idx in results:
        if not idx:
            continue
        keep = [i for i in range(len(st.residuals[k])) if i not in set(idx)]
        st.residuals[k] = [st.residuals[k][i] for i in keep]
        st.residual_emb[k] = st.residual_emb[k][keep] if keep else np.zeros((0, st.residual_emb[k].shape[1]))
        n += len(idx)
    return n


def rewrite_cells(llm, st: ProGraphStore, keys, turns, workers=6):
    date = turns[0].get("timestamp", "")
    conversation = "\n".join([f"[Session date: {date}]"] + [f"{t['speaker']}: {t['text']}" for t in turns])
    stats = {"rewrites": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _one(k):
        name = st.display_names.get(k, k)
        current = st.profiles.get(k, "No profile yet.")
        p0, c0 = llm.prompt_tokens, llm.completion_tokens
        resp = llm(_PROFILE_UPDATE_SYSTEM, _PROFILE_UPDATE_USER.format(entity_name=name, entity_type=st.types.get(k, "other"),
                                                                       current_profile=current, conversation=conversation), tag="prop:rewrite").strip()
        return k, _parse_profile_and_residuals(resp)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, keys))
    for k, (profile, residuals) in results:
        stats["rewrites"] += 1
        if profile and len(profile) > 20:
            st.profiles[k] = profile
            st.profile_emb[k] = st.embed_fn([profile])[0]
        if residuals:
            embs = st.embed_fn(residuals)
            for r, e in zip(residuals, embs):
                if not st._dup(k, e):
                    st.residuals.setdefault(k, []).append(r)
                    ex_ = st.residual_emb.get(k)
                    st.residual_emb[k] = e.reshape(1, -1) if ex_ is None else np.vstack([ex_, e.reshape(1, -1)])
    return stats


def rule_sets(st: ProGraphStore, inj, ce, llm, affected_keys):
    A, B = normalize(inj["A"]), normalize(inj["B"])
    conv = " ".join(t["text"] for t in inj["turns"])
    # R0: entities the released write path would touch: identification call + known-name scan
    from store import _IDENTIFY_CONCEPTS_SYSTEM, _IDENTIFY_CONCEPTS_USER, _parse_concepts
    raw = llm(_IDENTIFY_CONCEPTS_SYSTEM, _IDENTIFY_CONCEPTS_USER.format(conversation=conv), tag="prop:identify")
    r0 = {normalize(c["name"]) for c in _parse_concepts(raw) if normalize(c["name"]) in st.profiles}
    r0 |= {normalize(n) for n in st.find_names_in_text(conv) if normalize(n) in st.profiles}
    # R1: + cells whose text mentions both A and B
    both = {k for k in st.profiles if mentions(cell_text(st, k), inj["A"]) and mentions(cell_text(st, k), inj["B"])}
    r1 = r0 | both
    # R2: excitation over co-mention neighbourhood <= 2 hops from A,B; gate by cross-encoder on the change sentence
    frontier, seen = {A, B} & set(st.profiles), set()
    excited = set()
    for _ in range(2):
        nxt = set()
        for k in frontier:
            for n in st.find_names_in_text(cell_text(st, k)):
                nk = normalize(n)
                if nk in st.profiles and nk not in seen:
                    nxt.add(nk)
            # reverse direction: cells that mention k
            for c in st.profiles:
                if c not in seen and mentions(cell_text(st, c), st.display_names.get(k, k)):
                    nxt.add(c)
        seen |= frontier
        excited |= nxt
        frontier = nxt - seen
    excited -= {A, B}
    change = f"{inj['A']} and {inj['B']} are no longer {inj['relation']}; {inj['B']} moved to {inj['city']}."
    cand = sorted(excited)
    sc = ce.scores(change, [cell_text(st, k)[:2000] for k in cand]) if cand else []
    gated = {k for k, s in zip(cand, sc) if s > -2.0}  # ms-marco logits: > -2 ~ "plausibly related"
    r2 = r0 | gated
    r3 = r0 | set(affected_keys)
    return {"R0_prograph": r0, "R1_comention": r1, "R2_ca_gate": r2, "R3_oracle": r3, "R4_oracle_evict": r3}, {"excited": len(excited), "gated": len(gated), "both_mention": len(both)}


def run_scenario(sid, per_scenario, llm, embed_fn, ce, limit_cells):
    scenario, id2name, sessions, questions = load_bundle(sid)
    base = ProGraphStore.load(STORE_DIR / f"scenario_{sid:02d}", llm, embed_fn)
    out = {"scenario": sid, "injections": []}
    for idx in range(per_scenario):
        inj = make_injection(scenario, id2name, idx, sid)
        # candidate cells: anything mentioning A or B
        cands = [k for k in base.profiles if mentions(cell_text(base, k), inj["A"]) or mentions(cell_text(base, k), inj["B"])]
        cands = cands[:limit_cells] if limit_cells else cands
        with ThreadPoolExecutor(max_workers=6) as ex:
            pre = dict(zip(cands, ex.map(lambda k: judge_cell(llm, cell_text(base, k), inj), cands)))
        affected = [k for k, j in pre.items() if j["old"]]
        pre_resid = {}
        for k in affected:
            res = base.residuals.get(k, [])
            pre_resid[k] = judge_cell(llm, "\n".join(res), inj)["old"] if res else False
        rec = {"inj": {k: v for k, v in inj.items() if k != "turns"}, "n_candidates": len(cands), "n_affected": len(affected),
               "affected": affected, "rules": {}}
        sets, sizes = rule_sets(base, inj, ce, llm, affected)
        rec["sizes"] = sizes
        for rname, keys in sets.items():
            st = clone_store(base, llm, embed_fn)
            keys = sorted(k for k in keys if k in st.profiles)
            p0, c0 = llm.prompt_tokens, llm.completion_tokens
            stats = rewrite_cells(llm, st, keys, inj["turns"])
            if rname.endswith("_evict"):
                stats["evicted"] = evict_residuals(llm, st, keys, inj)
            stats["prompt_tokens"], stats["completion_tokens"] = llm.prompt_tokens - p0, llm.completion_tokens - c0
            # post-judge on affected cells (profile) and on rewritten non-affected cells (collateral)
            with ThreadPoolExecutor(max_workers=6) as ex:
                post = dict(zip(affected, ex.map(lambda k: judge_cell(llm, st.profiles.get(k, ""), inj), affected)))
            post_resid = {}
            for k in affected:
                res = st.residuals.get(k, [])
                post_resid[k] = judge_cell(llm, "\n".join(res), inj)["old"] if res else False
            rec["rules"][rname] = {
                "rewritten": keys, "n_rewritten": len(keys), **stats,
                "stale_profile": sum(1 for k in affected if post[k]["old"] and not post[k]["new"]),
                "new_fact": sum(1 for k in affected if post[k]["new"]),
                "stale_resid": sum(1 for k in affected if post_resid[k]),
                "collateral": sum(1 for k in keys if k not in affected),
                "missed": sum(1 for k in affected if k not in keys),
            }
            print(f"[s{sid} inj{idx} {rname}] affected={len(affected)} rewritten={len(keys)} stale={rec['rules'][rname]['stale_profile']} new={rec['rules'][rname]['new_fact']} tokens={stats['prompt_tokens']}/{stats['completion_tokens']}", flush=True)
        out["injections"].append(rec)
        json.dump(out, open(RESULTS_DIR / f"prop_s{sid:02d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def summarize():
    import glob
    files = sorted(glob.glob(str(RESULTS_DIR / "prop_s*.json")))
    rows = {}
    n_inj = 0; n_aff = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for rec in d["injections"]:
            n_inj += 1; n_aff += rec["n_affected"]
            for rname, r in rec["rules"].items():
                a = rows.setdefault(rname, {"rewrites": 0, "pt": 0, "ct": 0, "stale": 0, "new": 0, "stale_r": 0, "coll": 0, "missed": 0})
                a["rewrites"] += r["n_rewritten"]; a["pt"] += r["prompt_tokens"]; a["ct"] += r["completion_tokens"]
                a["stale"] += r["stale_profile"]; a["new"] += r["new_fact"]; a["stale_r"] += r["stale_resid"]; a["coll"] += r["collateral"]; a["missed"] += r["missed"]
    L = [f"# Write-path propagation ({n_inj} injections, {n_aff} affected cells in total)\n",
         "| rule | rewrites / injection | tokens / injection (in/out) | affected cells missed | profile still stale | profile has new fact | residual still stale | collateral rewrites |",
         "|---|---|---|---|---|---|---|---|"]
    for rname, a in rows.items():
        L.append(f"| {rname} | {a['rewrites']/n_inj:.1f} | {a['pt']/n_inj:.0f} / {a['ct']/n_inj:.0f} | {a['missed']}/{n_aff} | {a['stale']}/{n_aff} | {a['new']}/{n_aff} | {a['stale_r']}/{n_aff} | {a['coll']} |")
    (RESULTS_DIR / "prop_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="1,2,3")
    ap.add_argument("--per-scenario", type=int, default=4)
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summarize(); return
    from experiments_free import CE
    llm = LLM(usage_log=RESULTS_DIR / "usage.jsonl")
    embed_fn = make_embed_fn()
    ce = CE()
    for sid in [int(x) for x in a.scenarios.split(",")]:
        run_scenario(sid, a.per_scenario, llm, embed_fn, ce, a.limit_cells)
    ce.save()
    summarize()
    print("llm:", llm.stats())


if __name__ == "__main__":
    main()
