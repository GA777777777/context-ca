"""Answer-level check for the atomic (residual-only) store: does a context of the top-k residuals
by query cosine, with no narrative profiles at all, answer as well as the profile-based readouts?

  python atomic_answer.py --scenarios 1,2,3 --ks 80,160
Writes results/atomic_answer.json and prints accuracy by K with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data import load_bundle
from llm import LLM
from metrics import judge, n_tokens
from readpaths import _ANSWER_SYSTEM, _ANSWER_USER, _unit
from run_pilot import RESULTS_DIR, STORE_DIR, make_embed_fn
from store import ProGraphStore


def boot(vals, n=2000, seed=0):
    rng = random.Random(seed); vals = [float(v) for v in vals]
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    return m, bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="1,2,3")
    ap.add_argument("--ks", default="80,160")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    llm = LLM(usage_log=RESULTS_DIR / "usage.jsonl")
    embed_fn = make_embed_fn()
    recs = []
    for sid in [int(x) for x in a.scenarios.split(",")]:
        scenario, id2name, sessions, questions = load_bundle(sid)
        st = ProGraphStore.load(STORE_DIR / f"scenario_{sid:02d}", llm, embed_fn)
        atoms, owner, emb = [], [], []
        for k, res in st.residuals.items():
            for i, r in enumerate(res):
                atoms.append(r); owner.append(st.display_names.get(k, k)); emb.append(st.residual_emb[k][i])
        emb = np.vstack(emb); emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        jobs = []
        for q in questions:
            qv = _unit(embed_fn([q["question"]])[0])
            order = np.argsort(-(emb @ qv))
            for kk in ks:
                by_owner = defaultdict(list)
                for i in order[:kk]:
                    by_owner[owner[i]].append(atoms[i])
                ctx = "\n\n".join(f"=== {o} ===\nVerified Facts:" + "".join(f"\n- {r}" for r in rs) for o, rs in by_owner.items())
                jobs.append((sid, q, kk, ctx))

        def _run(job):
            sid_, q, kk, ctx = job
            pred = llm(_ANSWER_SYSTEM, _ANSWER_USER.format(context=ctx, question=q["question"]), tag="atomic:answer").strip()
            return {"sid": sid_, "qa_id": q["id"], "k": q["k"], "K": kk, "ctx_tokens": n_tokens(ctx), "predicted": pred,
                    "judge": judge(llm, q["question"], q["answer"], pred)}

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            recs += list(ex.map(_run, jobs))
        print(f"[s{sid}] done; llm={llm.stats()}", flush=True)
        json.dump(recs, open(RESULTS_DIR / "atomic_answer.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for kk in ks:
        rows = [r for r in recs if r["K"] == kk]
        m, lo, hi = boot([r["judge"] for r in rows])
        print(f"atom_cos_k{kk}: n={len(rows)} ctx_tokens={sum(r['ctx_tokens'] for r in rows)/len(rows):.0f} acc={m:.3f} [{lo:.3f}, {hi:.3f}] | "
              + " ".join(f"K={h}:{sum(r['judge'] for r in rows if r['k']==h)/max(1,sum(1 for r in rows if r['k']==h)):.2f}" for h in (1, 2, 3, 4, 5)))


if __name__ == "__main__":
    main()
