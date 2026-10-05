# arXiv submission checklist (v1)

## Files to upload
- `tex/main.tex` only (self-contained: manual `thebibliography`, no figures, no .bib, no .cls).
- Compiles with pdflatex / tectonic; packages used are all in TeX Live base: geometry, fontenc, inputenc,
  lmodern, microtype, booktabs, amsmath, amssymb, graphicx, array, enumitem, hyperref, cite.

## Before uploading
1. Fill in the author block in `md2tex.py` (`AUTHOR = ...`), re-run `python md2tex.py`, recompile.
2. Decide the code/data release URL and put it in the version line of `PAPER.md`
   ("Code, data mirror and per-question results: <URL>"). Currently says "to be released with the paper".
   Suggested: a public GitHub repo containing `pilot/` minus `.env`, `store/`, `results/usage.jsonl`.
   The MemHop mirror is MIT (ShengtongZhu/ProGraph); keep `LICENSE_ProGraph_MIT` next to it.
3. Metadata on arXiv:
   - Primary: cs.CL. Cross-list: cs.AI, cs.IR.
   - Title: as in the PDF.
   - Abstract: paste from `PAPER.md` (plain text; drop the markdown emphasis).
   - Comments: "26 pages, 6 tables. v1. Code and data to be released." (edit as needed)
   - License: CC BY 4.0 recommended (allows the later venue paper to reuse text with attribution).
4. New arXiv accounts in cs.* may need an endorsement; if asked, any arXiv author with recent cs.CL
   submissions can endorse via the link arXiv sends.

## Known cosmetic items (acceptable for v1)
- Table 1 is dense (8 wrapped columns, scriptsize); readable, but a landscape page would be nicer in v2.
- Tables 3, 5 and 6 are shrunk to text width.
- Section numbering is automatic; cross-references in the text ("Section 8.6") are literal and match.

## Regenerate
    python md2tex.py && wsl -e bash -lc 'cd /mnt/c/project/AIDD/paper_context_ca/tex && ~/bin/tectonic -X compile main.tex'
