"""Pilot: iterated threshold expansion (ProGraph Stage 2) vs one-shot selection,
scored at cell level against MemHop gold hops, with optional answer accuracy.

Usage
  python run_pilot.py --scenarios 1,2,3                  # build stores (LLM) + retrieval metrics (no LLM)
  python run_pilot.py --scenarios 1,2,3 --answer         # also answer + judge (LLM)
  python run_pilot.py --scenarios 1 --variants prograph,cosine_matched,crossenc_matched --answer
  python summarize.py                                    # aggregate results/*.json -> results/summary.md

Stores are cached under store/scenario_XX.{json,npz}; delete to rebuild.
Protocol: all sessions are ingested first, then every question is asked (not
interleaved as in ProGraph's harness); the store the read paths see is the final one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data import gold_chain, load_bundle
from llm import LLM
from metrics import hop_hits, judge, n_tokens, recall_at_k, step_first_hit
from readpaths import ReadPaths
from store import ProGraphStore

ROOT = Path(__file__).resolve().parent
STORE_DIR = ROOT / "store"
RESULTS_DIR = ROOT / "results"

K_GRID = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60, 80, 100, 140]

# variant grammar:
#   prograph[_t<tau>][_h<H>]          released Stage 2 (gate on best candidate, admit all)
#   percand[_t<tau>][_h<H>]           per-candidate gating (paper's description)
#   cosine_h0                         Stage 1 only (top-M)
#   cosine_k<k> | crossenc_k<k>       one-shot ranking, fixed k
#   cosine_matched | crossenc_matched one-shot, k = |S| of `prograph`
#   cosine_mpc | crossenc_mpc         one-shot, k = |S| of `percand`
#   laya_gate                         optional
DEFAULT_VARIANTS = "prograph,prograph_t0.3,prograph_t0.4,percand,percand_t0.3,percand_t0.4,cosine_h0," \
                   "cosine_k10,cosine_k20,cosine_k40,crossenc_k10,crossenc_k20,crossenc_k40,cosine_mpc,crossenc_mpc"
DEFAULT_ANSWER_VARIANTS = "prograph,percand,cosine_h0,cosine_k20,crossenc_k20,cosine_mpc,crossenc_mpc"


def _parse_ta(v: str, tau: float, H: int):
    import re
    m = re.search(r"_t([0-9.]+)", v)
    h = re.search(r"_h(\d+)", v)
    return (float(m.group(1)) if m else tau), (int(h.group(1)) if h else H)


def make_embed_fn(name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda texts: m.encode(texts, convert_to_numpy=True)


def build_or_load_store(sid: int, scenario, sessions, llm, embed_fn, force: bool = False, workers: int = 8) -> ProGraphStore:
    path = STORE_DIR / f"scenario_{sid:02d}"
    if path.with_suffix(".json").exists() and not force:
        s = ProGraphStore.load(path, llm, embed_fn)
        s.workers = workers
        if s.session_count >= len(sessions):
            print(f"[store] scenario {sid}: loaded ({len(s.profiles)} profiles, {sum(len(v) for v in s.residuals.values())} residuals)")
            return s
        print(f"[store] scenario {sid}: resuming from session {s.session_count}")
    else:
        known = [{"name": p["name"], "type": "person"} for p in scenario["people"]]
        s = ProGraphStore(llm, embed_fn, known_entities=known, workers=workers)
    for sess in sessions[s.session_count:]:
        t0 = time.time()
        st = s.ingest_session(sess["turns"])
        s.save(path)
        print(f"[store] scenario {sid}: session {sess['session_num']}/{len(sessions)} {st} ({time.time()-t0:.0f}s)")
    return s


def run_scenario(sid: int, variants, llm, embed_fn, do_answer: bool, force: bool, limit: int | None, M: int, tau: float, H: int, workers: int = 8, answer_workers: int = 3, answer_variants=None):
    scenario, id2name, sessions, questions = load_bundle(sid)
    store = build_or_load_store(sid, scenario, sessions, llm, embed_fn, force, workers)
    rp = ReadPaths(store, embed_fn)
    recs = []
    qs = questions[:limit] if limit else questions
    for qi, q in enumerate(qs):
        chain = gold_chain(q, id2name)
        rec = {"qa_id": q["id"], "k": q["k"], "question": q["question"], "gold": q["answer"],
               "gold_chain": [sorted(h["names"]) for h in chain], "variants": {}}
        base = rp.prograph(q["question"], M=M, tau=tau, H=H)
        base_pc = rp.prograph(q["question"], M=M, tau=tau, H=H, per_candidate=True)
        budget, budget_pc = len(base["selected"]), len(base_pc["selected"])
        rankings = {"cosine": rp.ranking(q["question"], "cosine"), "crossenc": rp.ranking(q["question"], "crossenc")}
        rec["recall_at_k"] = {m: {str(k): v for k, v in recall_at_k(r, chain, K_GRID).items()} for m, r in rankings.items()}
        rec["budgets"] = {"prograph": budget, "percand": budget_pc}
        outs = {}
        for v in variants:
            t_, h_ = _parse_ta(v, tau, H)
            if v == "prograph":
                outs[v] = base
            elif v == "percand":
                outs[v] = base_pc
            elif v.startswith("prograph"):
                outs[v] = rp.prograph(q["question"], M=M, tau=t_, H=h_)
            elif v.startswith("percand"):
                outs[v] = rp.prograph(q["question"], M=M, tau=t_, H=h_, per_candidate=True)
            elif v == "cosine_h0":
                outs[v] = rp.cosine_h0(q["question"], M=M)
            elif v in ("cosine_matched", "cosine_mpc"):
                outs[v] = rp.topk({k: -i for i, k in enumerate(rankings["cosine"])}, budget if v.endswith("matched") else budget_pc)
            elif v in ("crossenc_matched", "crossenc_mpc"):
                outs[v] = rp.topk({k: -i for i, k in enumerate(rankings["crossenc"])}, budget if v.endswith("matched") else budget_pc)
            elif v.startswith("cosine_k"):
                outs[v] = rp.topk({k: -i for i, k in enumerate(rankings["cosine"])}, int(v[len("cosine_k"):]))
            elif v.startswith("crossenc_k"):
                outs[v] = rp.topk({k: -i for i, k in enumerate(rankings["crossenc"])}, int(v[len("crossenc_k"):]))
            elif v == "laya_gate":
                try:
                    outs[v] = rp.laya_gate(q["question"], M=M, H=H)
                except ImportError:
                    continue
            else:
                raise ValueError(v)
        for v, o in outs.items():
            ctx = rp.build_context(q["question"], o["selected"])
            hits = hop_hits(o["selected"], chain)
            r = {"selected": o["selected"], "n_selected": len(o["selected"]), "steps": o["steps"],
                 "hop_hits": hits, "chain_recall": all(hits), "answer_hop_hit": hits[-1] if hits else None,
                 "first_hit_step": step_first_hit(o["trace"], chain), "ctx_tokens": n_tokens(ctx), "_ctx": ctx,
                 "trace_sizes": [len(t) for t in o["trace"]],
                 "trace_recall": [all(hop_hits(t, chain)) for t in o["trace"]]}
            rec["variants"][v] = r
        recs.append(rec)
        if (qi + 1) % 10 == 0:
            print(f"[s{sid:02d}] {qi+1}/{len(qs)} retrieved", flush=True)

    def _write(recs_):
        RESULTS_DIR.mkdir(exist_ok=True)
        out = {"scenario": sid, "params": {"M": M, "tau": tau, "H": H, "answer": do_answer},
               "store": {"profiles": len(store.profiles), "residuals": sum(len(v) for v in store.residuals.values())},
               "llm": llm.stats(), "records": [{**r, "variants": {v: {k: x for k, x in o.items() if k != "_ctx"} for v, o in r["variants"].items()}} for r in recs_]}
        json.dump(out, open(RESULTS_DIR / f"pilot_s{sid:02d}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    _write(recs)  # retrieval-level results are on disk before any answer call is made
    if do_answer:
        # One LLM answer + judge per distinct (question, selected set); variants that selected
        # the same cells share the result. Jobs are independent -> thread pool.
        from concurrent.futures import ThreadPoolExecutor
        jobs = {}
        for rec in recs:
            for v, r in rec["variants"].items():
                if answer_variants and v not in answer_variants:
                    continue
                jobs.setdefault((rec["qa_id"], tuple(r["selected"])), (rec, r["_ctx"]))

        def _job(item):
            (qa_id, key), (rec, ctx) = item
            try:
                pred = rp.answer(llm, rec["question"], ctx)
                return (qa_id, key), (pred, judge(llm, rec["question"], rec["gold"], pred), None)
            except Exception as e:  # noqa: BLE001
                return (qa_id, key), (None, None, str(e)[:200])

        with ThreadPoolExecutor(max_workers=min(workers, answer_workers)) as ex:
            done = dict(ex.map(_job, jobs.items()))
        n_err = 0
        for rec in recs:
            n_distinct = len({tuple(r["selected"]) for r in rec["variants"].values()})
            for v, r in rec["variants"].items():
                if (rec["qa_id"], tuple(r["selected"])) not in done:
                    continue
                pred, jd, err = done[(rec["qa_id"], tuple(r["selected"]))]
                r["predicted"], r["judge"] = pred, jd
                if err:
                    r["error"] = err
                    n_err += 1
                r["answer_shared"] = n_distinct < len(rec["variants"])
        print(f"[s{sid:02d}] answered {len(jobs)} distinct contexts for {len(recs)} questions x {len(variants)} variants; errors={n_err}", flush=True)
    _write(recs)
    print(f"[s{sid:02d}] written {RESULTS_DIR / f'pilot_s{sid:02d}.json'}  llm={llm.stats()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="1,2,3")
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--answer", action="store_true", help="generate answers and LLM-judge them")
    ap.add_argument("--answer-variants", default=DEFAULT_ANSWER_VARIANTS, help="subset of variants to answer (cost control)")
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N questions per scenario (smoke test)")
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8, help="parallel LLM calls on the write path")
    ap.add_argument("--answer-workers", type=int, default=3, help="parallel answer+judge calls (contexts can be large; mind TPM)")
    ap.add_argument("--build-only", action="store_true", help="only build/cache the stores, no retrieval")
    a = ap.parse_args()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    llm = LLM(usage_log=RESULTS_DIR / "usage.jsonl")
    print(f"[llm] backend={llm.backend} model={llm.model}")
    embed_fn = make_embed_fn()
    for sid in [int(x) for x in a.scenarios.split(",")]:
        if a.build_only:
            scenario, _, sessions, _ = load_bundle(sid)
            build_or_load_store(sid, scenario, sessions, llm, embed_fn, a.force_rebuild, a.workers)
            print(f"[s{sid:02d}] store ready; llm={llm.stats()}", flush=True)
            continue
        av = [x.strip() for x in a.answer_variants.split(",") if x.strip()]
        run_scenario(sid, variants, llm, embed_fn, a.answer, a.force_rebuild, a.limit, a.M, a.tau, a.H, a.workers, a.answer_workers, av)


if __name__ == "__main__":
    main()
