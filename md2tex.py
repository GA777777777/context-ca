"""Convert PAPER.md to arXiv-ready LaTeX (article class, manual bibliography).

  python md2tex.py  -> tex/main.tex
Handles: headings, paragraphs, bold/italic, inline code, [key; key] citations -> \\cite{},
pipe tables -> booktabs tabular (wide tables wrapped or shrunk), bullet/numbered lists,
footnotes written as lines starting with a superscript digit, references -> thebibliography.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

SRC = Path(__file__).with_name("PAPER.md")
OUT = Path(__file__).with_name("tex") / "main.tex"

AUTHOR = r"Gong ZhengQin\\ \small \texttt{zhengqingong@gmail.com}"

SPECIAL = {"\u00f8": r"{\o}", "\u00d8": r"{\O}", "\u0131": r"{\i}", "\u0142": r"{\l}", "\u00df": r"{\ss}", "\u00e6": r"{\ae}"}
ACCENTS = {"\u0301": "'", "\u0300": "`", "\u0308": '"', "\u0302": "^", "\u0303": "~", "\u030c": "v", "\u0327": "c", "\u0306": "u", "\u0304": "="}
SYMBOLS = {
    "\u2013": "--", "\u2014": "---", "\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''", "\u2026": r"\ldots{}",
    "\u2265": r"$\geq$", "\u2264": r"$\leq$", "\u00d7": r"$\times$", "\u2212": "--", "\u2192": r"$\rightarrow$", "\u2190": r"$\leftarrow$",
    "\u2016": r"$\Vert$", "\u2225": r"$\Vert$", "\u00f7": r"$\div$", "\u2227": r"$\wedge$", "\u2208": r"$\in$", "\u00b7": r"$\cdot$",
    "\u03c4": r"$\tau$", "\u03c3": r"$\sigma$", "\u03c1": r"$\rho$", "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$", "\u0394": r"$\Delta$",
    "\u00b2": r"$^2$", "\u2080": r"$_0$",
}


def deaccent(ch: str) -> str:
    if ch in SPECIAL:
        return SPECIAL[ch]
    d = unicodedata.normalize("NFD", ch)
    if len(d) == 2 and d[1] in ACCENTS:
        base = d[0]
        if base in "ij":
            base = "\\" + base
        return "{\\" + ACCENTS[d[1]] + "{" + base + "}}"
    return ch


def esc(t: str) -> str:
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        t = t.replace(a, b)
    t = t.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return t


def inline(t: str) -> str:
    codes, cites = [], []

    def _c(m):
        codes.append(m.group(1)); return f"@@CODE{len(codes) - 1}@@"

    def _cite(m):
        cites.append(",".join(k.strip() for k in m.group(1).split(";"))); return f"@@CITE{len(cites) - 1}@@"

    t = re.sub(r"`([^`]+)`", _c, t)
    t = re.sub(r"\[([a-z0-9\-]+(?:;\s*[a-z0-9\-]+)*)\]", _cite, t)
    # footnote markers: a superscript digit directly after punctuation (a power like |V|² is left alone)
    t = re.sub(r"(?<=[.,;:])([¹²³])", lambda m: "@@FN" + str("¹²³".index(m.group(1)) + 1) + "@@", t)
    t = esc(t)
    t = re.sub(r'"([^"\n]{1,200}?)"', r"``\1''", t)  # straight double quotes -> LaTeX quotes
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![\w\\])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\\emph{\1}", t)
    t = re.sub(r"\|V\|", r"$|V|$", t)
    t = re.sub(r"\|S\|", r"$|S|$", t)
    for a, b in SYMBOLS.items():
        t = t.replace(a, b)
    t = "".join(deaccent(ch) if ord(ch) > 127 else ch for ch in t)
    for i, c in enumerate(cites):
        t = t.replace(f"@@CITE{i}@@", r"\cite{" + c + "}")
    for i, c in enumerate(codes):
        t = t.replace(f"@@CODE{i}@@", r"\texttt{" + esc(c) + "}")
    return t


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|?\s*:?-{2,}", l)]
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    long_cells = max(len(c) for r in rows for c in r) > 40
    if ncol >= 7 and long_cells:  # wide prose table: wrapped ragged-right columns, tiny font
        w = (1.0 - 0.015 * ncol) / ncol
        colspec = "".join(">{\\raggedright\\arraybackslash}p{%.3f\\textwidth}" % w for _ in range(ncol))
        head = [r"\begin{table}[!tp]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{" + colspec + "}", r"\toprule"]
        tail = [r"\bottomrule", r"\end{tabular}", "@@CAPTION@@", r"\end{table}"]
    elif ncol >= 6:  # numeric wide table: shrink to text width
        head = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + "l" + "c" * (ncol - 1) + "}", r"\toprule"]
        tail = [r"\bottomrule", r"\end{tabular}}", "@@CAPTION@@", r"\end{table}"]
    else:
        head = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{" + "l" + "c" * (ncol - 1) + "}", r"\toprule"]
        tail = [r"\bottomrule", r"\end{tabular}", "@@CAPTION@@", r"\end{table}"]
    def cell(c):
        if ncol >= 7 and long_cells:  # narrow wrapped columns: let long compound tokens break
            c = c.replace("+", "+ ").replace("/", "/ ")
        return inline(c)
    out = list(head)
    out.append(" & ".join(cell(c) for c in rows[0]) + r" \\")
    out.append(r"\midrule")
    for r in rows[1:]:
        out.append(" & ".join(cell(c) for c in r) + r" \\")
    return out + tail


def main():
    src = SRC.read_text(encoding="utf-8")
    body, refs = src.split("\n## References", 1)
    lines = body.splitlines()
    title = lines[0].lstrip("# ").strip()
    tex, para, footnotes = [], [], {}
    pending_caption = None
    in_list = None

    def flush():
        nonlocal para
        if para:
            tex.append(inline(" ".join(para)))
            tex.append("")
            para = []

    def close_list():
        nonlocal in_list
        if in_list:
            tex.append(r"\end{" + in_list + "}")
            in_list = None

    i = 1
    while i < len(lines):
        l = lines[i]
        if l.startswith("*Version"):
            i += 1; continue
        if l.startswith("## Abstract"):
            flush(); tex.append(r"\begin{abstract}"); i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip():
                    tex.append(inline(lines[i].strip()))
                i += 1
            tex.append(r"\end{abstract}"); tex.append(""); continue
        m = re.match(r"^(#{2,3}) (?:\d+(?:\.\d+)?\s+)?(.+)$", l)
        if m:
            flush(); close_list()
            cmd = r"\section" if len(m.group(1)) == 2 else r"\subsection"
            tex.append(f"{cmd}{{{inline(m.group(2))}}}"); tex.append(""); i += 1; continue
        if l.startswith("|"):
            flush(); close_list()
            tl = []
            while i < len(lines) and lines[i].startswith("|"):
                tl.append(lines[i]); i += 1
            cap = pending_caption or ""
            pending_caption = None
            tex += [x.replace("@@CAPTION@@", (r"\caption{" + cap + "}") if cap else "") for x in table(tl)]
            tex.append(""); continue
        mt = re.match(r"^\*\*Table (\d+)\.\s*(.+?)\*\*\s*$", l)
        if mt:
            flush(); pending_caption = inline(mt.group(2)); i += 1; continue
        mf = re.match(r"^([\u00b9\u00b2\u00b3])\s+(.+)$", l)
        if mf:
            flush(); footnotes[mf.group(1)] = inline(mf.group(2)); i += 1; continue
        if re.match(r"^- ", l) or re.match(r"^\d+\. ", l):
            flush()
            kind = "itemize" if l.startswith("- ") else "enumerate"
            if in_list != kind:
                close_list(); tex.append(r"\begin{" + kind + "}"); in_list = kind
            tex.append(r"\item " + inline(re.sub(r"^(- |\d+\. )", "", l))); i += 1; continue
        if not l.strip():
            flush()
            if in_list and (i + 1 >= len(lines) or not re.match(r"^(- |\d+\. )", lines[i + 1])):
                close_list(); tex.append("")
            i += 1; continue
        if l.startswith("<!--"):
            i += 1; continue
        para.append(l.strip()); i += 1
    flush(); close_list()
    out = "\n".join(tex)
    for mark, txt in footnotes.items():
        out = out.replace("@@FN" + str("¹²³".index(mark) + 1) + "@@", r"\footnote{" + txt + "}", 1)
    bib = [r"\begin{thebibliography}{99}"]
    for l in refs.splitlines():
        m = re.match(r"^- \[([a-z0-9\-]+)\] (.+)$", l)
        if m:
            bib.append(r"\bibitem{" + m.group(1) + "} " + inline(m.group(2)))
    bib.append(r"\end{thebibliography}")
    head = "\n".join([
        r"\documentclass[11pt]{article}", r"\usepackage[margin=1in]{geometry}", r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}", r"\usepackage{lmodern}", r"\usepackage{microtype}", r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}", r"\usepackage{graphicx}", r"\usepackage{array}", r"\usepackage{enumitem}", r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{cite}", r"\setlist{nosep}",
        r"\title{" + inline(title) + "}", r"\author{" + AUTHOR + "}", r"\date{October 2026}",
        r"\begin{document}", r"\maketitle", "",
    ])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(head + out + "\n\n" + "\n".join(bib) + "\n\\end{document}\n", encoding="utf-8")
    nonascii = sorted({ch for ch in out + "".join(bib) if ord(ch) > 127})
    print("wrote", OUT, "| sections:", out.count(r"\section{"), "| tables:", out.count(r"\begin{table}"), "| footnotes:", out.count(r"\footnote{"),
          "| cites:", len(re.findall(r"\\cite\{", out)), "| bibitems:", len(bib) - 2, "| non-ascii left:", nonascii)


if __name__ == "__main__":
    main()
