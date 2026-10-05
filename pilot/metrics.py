"""Cell-level and answer-level metrics."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from data import cell_matches_name

JUDGE_SYSTEM = "You are a precise answer evaluator. Output only 'correct' or 'incorrect'."
JUDGE_PROMPT = """Determine if the predicted answer is semantically correct compared to the gold answer.
They do not need to match word-for-word — synonyms, rephrasings, and equivalent expressions all count as correct.
Focus on whether the core factual information matches.

Output ONLY "correct" or "incorrect".

Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted}"""

_enc = None


def n_tokens(text: str) -> int:
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text or ""))


def hop_hits(selected: List[str], chain: List[Dict[str, Any]]) -> List[bool]:
    """For each gold hop, True if any of its gold names is covered by a selected cell."""
    out = []
    for h in chain:
        names = h["names"]
        out.append(any(cell_matches_name(c, n) for c in selected for n in names) if names else True)
    return out


def step_first_hit(trace: List[List[str]], chain: List[Dict[str, Any]]) -> List[int]:
    """Step index at which each gold hop first became covered (-1 if never)."""
    out = []
    for h in chain:
        found = -1
        for t, sel in enumerate(trace):
            if any(cell_matches_name(c, n) for c in sel for n in h["names"]):
                found = t
                break
        out.append(found)
    return out


def recall_at_k(ranking: List[str], chain: List[Dict[str, Any]], ks: List[int]) -> Dict[int, bool]:
    """Full-chain recall when taking the top-k of a ranking, for each k."""
    out = {}
    for k in ks:
        out[k] = all(hop_hits(ranking[:k], chain))
    return out


def judge(llm, question: str, gold: str, predicted: str) -> float:
    resp = llm(JUDGE_SYSTEM, JUDGE_PROMPT.format(question=question, gold_answer=gold, predicted=predicted), tag="judge")
    s = resp.strip().lower()
    if "incorrect" in s:
        return 0.0
    if "correct" in s:
        return 1.0
    return 0.0
