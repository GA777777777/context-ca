"""Re-run answer + judge for records whose answer call failed (429/400), sequentially
and paced to stay under the provider's tokens-per-minute limit. Updates results in place.

  python retry_errors.py --scenarios 1,2,3 --tpm 450000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm import LLM
from metrics import judge
from readpaths import ReadPaths
from run_pilot import RESULTS_DIR, STORE_DIR, make_embed_fn
from store import ProGraphStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="1,2,3")
    ap.add_argument("--tpm", type=int, default=450000, help="token budget per minute to pace against")
    a = ap.parse_args()
    llm = LLM(usage_log=RESULTS_DIR / "usage.jsonl")
    embed_fn = make_embed_fn()
    for sid in [int(x) for x in a.scenarios.split(",")]:
        path = RESULTS_DIR / f"pilot_s{sid:02d}.json"
        d = json.load(open(path, encoding="utf-8"))
        store = ProGraphStore.load(STORE_DIR / f"scenario_{sid:02d}", llm, embed_fn)
        rp = ReadPaths(store, embed_fn)
        todo = [(r, v, o) for r in d["records"] for v, o in r["variants"].items() if o.get("error")]
        print(f"[s{sid:02d}] {len(todo)} failed answers to retry", flush=True)
        cache = {}
        window_start, window_tokens = time.time(), 0
        fixed = 0
        for i, (r, v, o) in enumerate(todo):
            key = (r["qa_id"], tuple(o["selected"]))
            if key not in cache:
                ctx = rp.build_context(r["question"], o["selected"])
                need = o["ctx_tokens"] + 500
                # pace: if this call would exceed the per-minute budget, wait for the window to roll
                if window_tokens + need > a.tpm:
                    sleep_for = max(0.0, 60.0 - (time.time() - window_start)) + 1.0
                    time.sleep(sleep_for)
                    window_start, window_tokens = time.time(), 0
                elif time.time() - window_start > 60:
                    window_start, window_tokens = time.time(), 0
                try:
                    pred = rp.answer(llm, r["question"], ctx)
                    cache[key] = (pred, judge(llm, r["question"], r["gold"], pred), None)
                except Exception as e:  # noqa: BLE001
                    cache[key] = (None, None, str(e)[:200])
                window_tokens += need
            pred, jd, err = cache[key]
            if err:
                o["error"] = err
            else:
                o.pop("error", None)
                o["predicted"], o["judge"] = pred, jd
                fixed += 1
            if (i + 1) % 10 == 0:
                print(f"[s{sid:02d}] {i+1}/{len(todo)} retried, fixed={fixed}", flush=True)
        d["llm_retry"] = llm.stats()
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[s{sid:02d}] fixed {fixed}/{len(todo)}; llm={llm.stats()}", flush=True)


if __name__ == "__main__":
    main()
