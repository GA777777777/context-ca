# Context as a Cellular Automaton

Code, data and results for

> Gong ZhengQin. *Context as a Cellular Automaton: A Survey of Local-Rule Memory and Context Management for LLM Agents.* 2026. Zenodo, v1: https://doi.org/10.5281/zenodo.23176958 (all versions: https://doi.org/10.5281/zenodo.23176957). arXiv version pending endorsement.

The paper reads LLM-agent memory systems as local-rule dynamical processes on a graph of memory
units (a Context-CA) and tests, on MemHop, whether iterated neighbourhood expansion beats one-shot
selection, where the neighbourhood still matters (edge text, write-path propagation), and what a
trained transition rule would have to beat.

## Layout

| path | what |
|---|---|
| `PAPER.md` | the paper (source of truth); `md2tex.py` renders it to `tex/main.tex` |
| `tex/main.pdf` | compiled PDF |
| `REFS_VERIFIED.md` | every reference checked against its primary page, with corrections |
| `PILOT_RESULTS.md`, `FREE_RESULTS.md`, `PROP_RESULTS.md` | result tables (Sections 8.2, 8.4/8.6, 8.5) |
| `pilot/` | all experiment code; see `pilot/README.md` |
| `pilot/data/memhop/` | mirror of the MemHop benchmark (MIT, from ShengtongZhu/ProGraph) |
| `pilot/store/` | the three ProGraph stores the experiments ran on (profiles, residuals, embeddings) |
| `pilot/results/` | per-question JSON for every experiment |

## Reproduce

```bash
cd pilot
pip install -r requirements.txt
cp .env.example .env          # only needed for the LLM-calling scripts
python run_pilot.py --scenarios 1,2,3 --answer     # Section 8.2 (uses cached stores; LLM for answers)
python experiments_free.py all                     # Sections 8.4 and 8.6 (no LLM; local models)
python write_propagation.py --scenarios 1,2,3      # Section 8.5 (LLM)
python atomic_answer.py --scenarios 1,2,3          # Section 8.4 answer-level check (LLM)
python analyze.py; python summarize.py; python experiments_free.py summary
```

Stores are cached, so the retrieval-level experiments run without any API key. Building the
stores from scratch (`--force-rebuild`) and the answer/judge steps call the model named in `.env`.

## Render the paper

```bash
python md2tex.py && cd tex && tectonic -X compile main.tex   # or pdflatex main.tex twice
```

## License

Code: MIT. MemHop data: MIT (see `pilot/data/memhop/LICENSE_ProGraph_MIT`). Paper text: CC BY 4.0.
