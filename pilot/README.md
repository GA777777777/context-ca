# Pilot: iterated threshold expansion vs one-shot selection on MemHop

The falsifier from Section 7.2 / 9 of the survey: does ProGraph's iterated Stage 2
expansion select better cells than a one-shot ranker given the same budget?

## What it does

1. **Write path (LLM, cached).** Builds the ProGraph store (entity profiles +
   compression residuals) for each scenario with the *verbatim* ProGraph prompts
   (`store.py`, MIT-licensed code from github.com/ShengtongZhu/ProGraph), using an
   Azure / OpenAI-compatible client that logs token usage. Cached under `store/`.
2. **Read paths (no LLM).** For every question, runs several selection rules over the
   same store and records the selected cells, the per-step trace, and the context size:
   - `prograph` — Stage 1 (top-M cosine + name boost) then Stage 2 expansion as released
     (gate on the best candidate, admit all), H steps
   - `prograph_percand` — Stage 2 with per-candidate gating (the paper's description)
   - `cosine_h0` — Stage 1 only (H = 0)
   - `cosine_matched` / `crossenc_matched` — one-shot ranking of *all* cells by cosine or
     by a cross-encoder, taking exactly |S| cells where |S| is what `prograph` selected
     for that question (same-budget control)
   - `cosine_k5`, `crossenc_k5`, ... — fixed-k one-shot controls
   - `laya_gate` — optional: expansion gated by a Laya `noul` question per candidate
3. **Cell-level labels.** Each MemHop question's `decomposition` gives one evidence
   observation per hop; the observation's `person_id` is the gold entity for that hop.
   Metrics: per-hop coverage, full-chain recall, answer-hop recall, step at which each
   hop was first covered, |S|, context tokens.
4. **Answer level (optional, LLM).** `--answer` generates an answer from each variant's
   context with the ProGraph answer prompt and scores it with the ProGraph judge prompt.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in Azure endpoint / key / deployment
python run_pilot.py --scenarios 1 --limit 10           # smoke test (builds store for scenario 1)
python run_pilot.py --scenarios 1,2,3                  # retrieval metrics, 300 questions
python run_pilot.py --scenarios 1,2,3 --answer         # + answers and judge
python summarize.py                                    # -> results/summary.md
```

Cost: the write path is roughly 20 sessions x (1 + entities) calls per scenario; answers
and judge add 2 calls per question per variant. Exact numbers land in `results/usage.jsonl`.

## Deviations from the ProGraph harness (state these in the paper)

- All sessions are ingested before any question is asked (ProGraph interleaves).
- Embedder is the same (all-MiniLM-L6-v2); the LLM is whatever the `.env` points at, not gpt-4o-mini.
- The store is built once and shared by all read-path variants, so differences are due
  to the read path alone.
- Gold-name to cell matching accepts either the full name or its first token, because
  ProGraph may register "Alice" and "Alice Chen" as separate cells.

## Files

- `llm.py` client + usage log; `data.py` loader + gold chains; `store.py` write path;
  `readpaths.py` variants + readout; `metrics.py`; `run_pilot.py`; `summarize.py`.
- `data/memhop/` mirror of the benchmark (MIT, see `LICENSE_ProGraph_MIT`).
