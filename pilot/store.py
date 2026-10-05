"""ProGraph write path (profiles + compression residuals), vendored from
github.com/ShengtongZhu/ProGraph prograph/runners/prograph_runner.py (MIT), with:
  * an injectable LLM client (Azure / OpenAI-compatible) instead of an OpenAI key,
  * per-session checkpointing of the store to disk,
  * exposure of the internal state so read paths can be instrumented.
Prompts are verbatim so that the store is the same object the paper evaluates.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data import normalize

_IDENTIFY_CONCEPTS_SYSTEM = """Given this conversation, list all important entities discussed.
Include: person names, organizations, places, events, pets, projects — anything worth tracking.
Return a JSON array of objects: [{"name": "...", "type": "person|org|place|event|other"}]
Only include entities with substantial information in the conversation."""

_IDENTIFY_CONCEPTS_USER = """Conversation:
{conversation}

Entities (JSON array):"""

_PROFILE_UPDATE_SYSTEM = """You are maintaining a knowledge profile for a specific entity.
The conversation input may begin with a "[Session date: ...]" line giving the absolute date
when the conversation took place. Use this session date to resolve any relative time
expressions ("yesterday", "last week", "next month", "two days ago", "last Sunday", etc.)
into absolute dates. Always store the resolved absolute date in both the profile and residuals,
never the relative phrase.

Do TWO things in one response:

1. PROFILE: Update the narrative profile to incorporate new information.
   - Include ALL known facts, relationships, attributes, and events.
   - Keep it concise but comprehensive. Preserve existing info unless contradicted.
   - Write in third person.
   - Convert relative dates to absolute dates using the session date.

2. RESIDUALS: List specific facts from the NEW conversation that a summary
   might simplify or lose. Include ANY of these:
   - Exact dates or times, ABSOLUTE form only (e.g., "7 May 2023", "the Sunday before 25 May 2023")
     Never write "yesterday" / "last week" / "next month"; compute the date from the session date.
   - Specific numbers, counts, or quantities (e.g., "3 children", "visited 2 times", "$85K")
   - Proper names of things: book titles, song/album names, band/artist names, artwork
     titles, restaurant names, event names (e.g., 'read "Becoming Nicole"')
   - Identity and personal status: gender identity, relationship status, dietary
     restrictions, disabilities (e.g., "transgender woman", "single", "vegetarian")
   - Specific items in a list: instruments played, hobbies practiced, places visited,
     types of objects made (e.g., "made bowls and a cup at pottery class")
   - Cross-person facts: recommendations, gifts, shared experiences
     (e.g., "Melanie read 'Becoming Nicole' on Caroline's recommendation")
   - State changes: before/after transitions (e.g., "quit TechCorp, joined Google")
   Each residual must be a self-contained statement including the entity name.
   If no such facts exist in this conversation, write NONE.

OUTPUT FORMAT (follow exactly):
PROFILE:
[updated profile text]

RESIDUALS:
- [fact 1]
- [fact 2]"""

_PROFILE_UPDATE_USER = """Entity: {entity_name} (type: {entity_type})

Current profile:
{current_profile}

New conversation:
{conversation}

Output (PROFILE then RESIDUALS):"""


def _parse_concepts(response: str) -> List[Dict[str, str]]:
    m = re.search(r"\[.*\]", response, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for x in arr:
        if isinstance(x, dict) and x.get("name"):
            out.append({"name": str(x["name"]).strip(), "type": str(x.get("type", "other"))})
    return out


def _parse_profile_and_residuals(response: str) -> Tuple[str, List[str]]:
    m = re.search(r"\nRESIDUALS?\s*:", response, re.IGNORECASE)
    if not m:
        prof = re.sub(r"^\s*PROFILE\s*:\s*", "", response, flags=re.IGNORECASE).strip()
        return prof, []
    profile_part = re.sub(r"^\s*PROFILE\s*:\s*", "", response[: m.start()], flags=re.IGNORECASE).strip()
    residuals_part = response[m.end():].strip()
    if not residuals_part or residuals_part.upper().startswith("NONE"):
        return profile_part, []
    residuals = []
    for line in residuals_part.split("\n"):
        line = line.strip()
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()
            if line:
                residuals.append(line)
    return profile_part, residuals


class ProGraphStore:
    def __init__(self, llm, embed_fn, known_entities: List[Dict[str, str]] | None = None,
                 residual_dedup_threshold: float = 0.9, workers: int = 8):
        self.llm = llm
        self.embed_fn = embed_fn
        self.workers = workers
        self.residual_dedup_threshold = residual_dedup_threshold
        self.profiles: Dict[str, str] = {}
        self.display_names: Dict[str, str] = {}
        self.types: Dict[str, str] = {}
        self.profile_emb: Dict[str, np.ndarray] = {}
        self.residuals: Dict[str, List[str]] = {}
        self.residual_emb: Dict[str, np.ndarray] = {}
        self.known_names: List[str] = []
        self.session_count = 0
        for e in known_entities or []:
            n = e.get("name", "")
            self.display_names[normalize(n)] = n
            self.types[normalize(n)] = e.get("type", "person")
            self.known_names.append(n)

    # ---- name matching (verbatim semantics) ----
    def find_names_in_text(self, text: str) -> List[str]:
        tl = text.lower()
        out = []
        for name in self.known_names:
            nl = name.lower()
            if nl in tl:
                out.append(name)
                continue
            first = nl.split()[0] if nl.split() else nl
            if len(first) > 2 and re.search(r"\b" + re.escape(first) + r"\b", tl):
                out.append(name)
        return out

    # ---- write path ----
    def ingest_session(self, turns: List[Dict[str, Any]]) -> Dict[str, int]:
        self.session_count += 1
        if not turns:
            return {"entities": 0}
        date = turns[0].get("timestamp", "")
        lines = [f"[Session date: {date}]"] if date else []
        lines += [f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns]
        conversation = "\n".join(lines)
        raw = self.llm(_IDENTIFY_CONCEPTS_SYSTEM, _IDENTIFY_CONCEPTS_USER.format(conversation=conversation), tag="write:identify")
        concepts = _parse_concepts(raw)
        for name in self.find_names_in_text(conversation):
            if not any(normalize(c["name"]) == normalize(name) for c in concepts):
                concepts.append({"name": name, "type": self.types.get(normalize(name), "person")})
        n = 0
        todo = []
        seen = set()
        for c in concepts:
            name = c["name"].strip()
            if len(name) < 2:
                continue
            key = normalize(name)
            if key in seen:
                continue
            seen.add(key)
            if name not in self.known_names:
                self.known_names.append(name)
            self.display_names[key] = name
            self.types[key] = c.get("type", "other")
            todo.append((key, name, c.get("type", "other"), self.profiles.get(key, "No profile yet.")))

        def _call(item):
            key, name, ctype, current = item
            resp = self.llm(_PROFILE_UPDATE_SYSTEM, _PROFILE_UPDATE_USER.format(
                entity_name=name, entity_type=ctype, current_profile=current, conversation=conversation),
                tag="write:profile").strip()
            return key, _parse_profile_and_residuals(resp)

        # Per-entity calls are independent (each reads only its own profile), so run them in parallel.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(_call, todo))
        for key, (profile, residuals) in results:
            n += 1
            if profile and len(profile) > 20:
                self.profiles[key] = profile
                self.profile_emb[key] = self.embed_fn([profile])[0]
            if residuals:
                embs = self.embed_fn(residuals)
                for r, e in zip(residuals, embs):
                    if not self._dup(key, e):
                        self.residuals.setdefault(key, []).append(r)
                        ex = self.residual_emb.get(key)
                        self.residual_emb[key] = e.reshape(1, -1) if ex is None else np.vstack([ex, e.reshape(1, -1)])
        return {"entities": n, "profiles": len(self.profiles), "residuals": sum(len(v) for v in self.residuals.values())}

    def _dup(self, key: str, e: np.ndarray) -> bool:
        ex = self.residual_emb.get(key)
        if ex is None or len(ex) == 0:
            return False
        en = e / (np.linalg.norm(e) + 1e-9)
        xn = ex / (np.linalg.norm(ex, axis=1, keepdims=True) + 1e-9)
        return float(np.max(xn @ en)) > self.residual_dedup_threshold

    # ---- persistence ----
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "profiles": self.profiles, "display_names": self.display_names, "types": self.types,
            "residuals": self.residuals, "known_names": self.known_names, "session_count": self.session_count,
        }
        json.dump(meta, open(path.with_suffix(".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        arrs = {f"p::{k}": v for k, v in self.profile_emb.items()}
        arrs.update({f"r::{k}": v for k, v in self.residual_emb.items()})
        np.savez(path.with_suffix(".npz"), **arrs)

    @classmethod
    def load(cls, path: Path, llm=None, embed_fn=None) -> "ProGraphStore":
        path = Path(path)
        meta = json.load(open(path.with_suffix(".json"), encoding="utf-8"))
        s = cls(llm, embed_fn)
        s.profiles, s.display_names, s.types = meta["profiles"], meta["display_names"], meta["types"]
        s.residuals, s.known_names, s.session_count = meta["residuals"], meta["known_names"], meta["session_count"]
        z = np.load(path.with_suffix(".npz"))
        for k in z.files:
            kind, key = k.split("::", 1)
            (s.profile_emb if kind == "p" else s.residual_emb)[key] = z[k]
        return s
