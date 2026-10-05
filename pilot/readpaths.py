"""Read-path variants over a ProGraphStore, each returning the selected cell set
with a per-step trace so the pilot can score cell-level recall against gold hops.

Variants
  prograph          : ProGraph Stage 1 + Stage 2 as released (gate on the best candidate,
                      then admit ALL candidates of the hop)          — iterated, H steps
  prograph_percand  : same, but each candidate must clear the gate itself
                      (the rule as described in the paper's worked example)
  cosine_h0         : Stage 1 only (top-M by cosine + name boost)    — H = 0 control
  cosine_k          : one-shot cosine ranking over all cells, top-k
  crossenc_k        : one-shot cross-encoder (query, profile) ranking, top-k
  *_matched         : k set per query to |S| of the `prograph` run (same-budget control)
  laya_gate         : optional; Stage 1 seeds, expansion gated by a Laya `noul` question
                      per candidate (step 2 of the paper's agenda). Requires `laya`.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np

from data import normalize

_ANSWER_SYSTEM = """Answer the question using the entity information below.
Each entity has a Profile (narrative summary) and Verified Facts (precise details).
IMPORTANT: Verified Facts are extracted directly from conversations and override
the profile when they conflict. Always check Verified Facts first.
Give a short, specific answer (a name, date, number, or fact).
If you cannot answer, say "I don't know"."""

_ANSWER_USER = """{context}

Question: {question}

Answer:"""

MAX_RESIDUALS_PER_CELL = 8


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


class ReadPaths:
    def __init__(self, store, embed_fn, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.s = store
        self.embed_fn = embed_fn
        self._ce = None
        self._ce_name = cross_encoder_name
        self._laya = None

    # ---- scoring primitives ----
    def cosine_scores(self, query: str) -> Dict[str, float]:
        q = _unit(self.embed_fn([query])[0])
        return {k: float(np.dot(q, _unit(e))) for k, e in self.s.profile_emb.items()}

    def stage1(self, query: str, M: int = 5, boost: float = 0.3) -> tuple[Set[str], Dict[str, float]]:
        scores = self.cosine_scores(query)
        for name in self.s.find_names_in_text(query):
            k = normalize(name)
            if k in scores:
                scores[k] += boost
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return {k for k, _ in ranked[:M]}, scores

    def cross_encoder_scores(self, query: str) -> Dict[str, float]:
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self._ce_name)
        keys = [k for k in self.s.profiles]
        pairs = [(query, self.s.profiles[k]) for k in keys]
        sc = self._ce.predict(pairs, batch_size=32)
        return {k: float(v) for k, v in zip(keys, sc)}

    # ---- variants ----
    def prograph(self, query: str, M: int = 5, tau: float = 0.2, H: int = 5, per_candidate: bool = False) -> Dict[str, Any]:
        selected, scores = self.stage1(query, M)
        trace = [sorted(selected)]
        qv = _unit(self.embed_fn([query])[0])
        for _ in range(H):
            cand: Set[str] = set()
            best = 0.0
            cand_scores = {}
            for k in selected:
                for name in self.s.find_names_in_text(self.s.profiles.get(k, "")):
                    n = normalize(name)
                    if n in self.s.profiles and n not in selected:
                        e = self.s.profile_emb.get(n)
                        if e is not None:
                            sc = float(np.dot(qv, _unit(e)))
                            cand_scores[n] = sc
                            best = max(best, sc)
                        cand.add(n)
            if not cand or best < tau:
                break
            if per_candidate:
                add = {n for n in cand if cand_scores.get(n, -1.0) >= tau}
                if not add:
                    break
                selected |= add
            else:
                selected |= cand
            trace.append(sorted(selected))
        return {"selected": sorted(selected), "trace": trace, "steps": len(trace) - 1}

    def cosine_h0(self, query: str, M: int = 5) -> Dict[str, Any]:
        sel, _ = self.stage1(query, M)
        return {"selected": sorted(sel), "trace": [sorted(sel)], "steps": 0}

    def topk(self, scores: Dict[str, float], k: int) -> Dict[str, Any]:
        ranked = sorted(scores.items(), key=lambda x: -x[1])[: max(1, k)]
        sel = sorted(kk for kk, _ in ranked)
        return {"selected": sel, "trace": [sel], "steps": 0}

    def cosine_k(self, query: str, k: int, boost: float = 0.3) -> Dict[str, Any]:
        _, scores = self.stage1(query, M=len(self.s.profiles), boost=boost)
        return self.topk(scores, k)

    def crossenc_k(self, query: str, k: int) -> Dict[str, Any]:
        return self.topk(self.cross_encoder_scores(query), k)

    def ranking(self, query: str, method: str, boost: float = 0.3) -> List[str]:
        """Full ordering of all cells by a one-shot scorer (for recall@k curves)."""
        if method == "cosine":
            _, scores = self.stage1(query, M=len(self.s.profiles), boost=boost)
        elif method == "crossenc":
            scores = self.cross_encoder_scores(query)
        else:
            raise ValueError(method)
        return [k for k, _ in sorted(scores.items(), key=lambda x: -x[1])]

    def laya_gate(self, query: str, M: int = 5, theta: float = 0.5, H: int = 5, model: Optional[str] = None) -> Dict[str, Any]:
        """Expansion gated by a Laya noul question per candidate. Optional dependency."""
        if self._laya is None:
            from laya import Router  # type: ignore
            self._laya = Router()
        selected, _ = self.stage1(query, M)
        trace = [sorted(selected)]
        for _ in range(H):
            cand: Set[str] = set()
            for k in selected:
                for name in self.s.find_names_in_text(self.s.profiles.get(k, "")):
                    n = normalize(name)
                    if n in self.s.profiles and n not in selected:
                        cand.add(n)
            if not cand:
                break
            add = set()
            for n in cand:
                state = f"Question: {query}\n\nEntity profile:\n{self.s.profiles[n][:1500]}"
                qs = {"rel": {"type": "noul", "instructions": "Does this entity's profile contain the answer to the question, or a fact that links to the entity that has the answer?"}}
                r = self._laya.predict(state, qs, model=model) if model else self._laya.predict(state, qs)
                p = float(r["answers"]["rel"]["noul"])
                if p >= theta:
                    add.add(n)
            if not add:
                break
            selected |= add
            trace.append(sorted(selected))
        return {"selected": sorted(selected), "trace": trace, "steps": len(trace) - 1}

    # ---- readout ----
    def build_context(self, query: str, selected: List[str]) -> str:
        qv = _unit(self.embed_fn([query])[0])
        parts = []
        for k in selected:
            prof = self.s.profiles.get(k, "")
            if not prof:
                continue
            part = f"=== {self.s.display_names.get(k, k)} ===\n{prof}"
            res = self.s.residuals.get(k, [])
            re_ = self.s.residual_emb.get(k)
            if res and re_ is not None and len(re_) > 0:
                rn = re_ / (np.linalg.norm(re_, axis=1, keepdims=True) + 1e-9)
                idx = np.argsort(-(rn @ qv).flatten())[:MAX_RESIDUALS_PER_CELL]
                chosen = [res[i] for i in idx if i < len(res)]
                if chosen:
                    part += "\nVerified Facts:" + "".join(f"\n- {r}" for r in chosen)
            parts.append(part)
        return "\n\n".join(parts)

    def answer(self, llm, query: str, context: str) -> str:
        if not context:
            return "I don't know"
        return llm(_ANSWER_SYSTEM, _ANSWER_USER.format(context=context, question=query), tag="read:answer").strip()
