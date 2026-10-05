"""MemHop loading and gold-chain extraction.

Data files come from github.com/ShengtongZhu/ProGraph (MIT), mirrored under data/memhop/.
A question's gold chain is derived from its `decomposition` (one entry per hop, each
pointing at an evidence observation) and the `evidence` list (which names the person
each observation belongs to). This is the cell-level label used by the pilot.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "memhop"

SCENARIO_FILES = {
    1: "scenario_01_university_friends.json", 2: "scenario_02_tech_startup.json",
    3: "scenario_03_community_garden.json", 4: "scenario_04_sports_league.json",
    5: "scenario_05_music_conservatory.json", 6: "scenario_06_hospital_staff.json",
    7: "scenario_07_art_collective.json", 8: "scenario_08_cooking_club.json",
    9: "scenario_09_adventure_group.json", 10: "scenario_10_parent_group.json",
}


def normalize(name: str) -> str:
    return name.strip().lower()


def load_bundle(sid: int):
    scenario = json.load(open(DATA / "scenarios" / SCENARIO_FILES[sid], encoding="utf-8"))
    raw = json.load(open(DATA / "conversations" / f"scenario_{sid:02d}_sessions.json", encoding="utf-8"))
    qs = json.load(open(DATA / "questions" / f"scenario_{sid:02d}_questions.json", encoding="utf-8"))
    id2name = {p["person_id"]: p["name"] for p in scenario["people"]}
    sessions = []
    for i, s in enumerate(raw.get("sessions", raw)):
        turns = [{
            "speaker": id2name.get(t["speaker"], t["speaker"]),
            "text": t["text"],
            "session_num": i + 1,
            "timestamp": s.get("date", ""),
        } for t in s.get("turns", [])]
        sessions.append({"session_num": i + 1, "date": s.get("date", ""), "turns": turns})
    questions = qs.get("questions", qs)
    return scenario, id2name, sessions, questions


def gold_chain(q: Dict[str, Any], id2name: Dict[str, str]) -> List[Dict[str, Any]]:
    """Per-hop gold entities: [{hop, person_ids, names}] ordered by hop."""
    ev = {e["obs_id"]: e for e in q.get("evidence", [])}
    hops = []
    for d in sorted(q.get("decomposition", []), key=lambda x: x.get("hop", 0)):
        e = ev.get(d.get("evidence_id"))
        pids = {e["person_id"]} if e else set()
        hops.append({"hop": d.get("hop"), "person_ids": pids, "names": {id2name[p] for p in pids if p in id2name}})
    return hops


def gold_entities(q: Dict[str, Any], id2name: Dict[str, str]) -> set:
    return {n for h in gold_chain(q, id2name) for n in h["names"]}


def cell_matches_name(cell_key: str, full_name: str) -> bool:
    """A store cell (keyed by normalized name as the LLM identified it) matches a gold
    full name if it equals the full name or its first token (ProGraph may register
    'Alice' and 'Alice Chen' as separate cells)."""
    k = normalize(cell_key)
    n = normalize(full_name)
    first = n.split()[0] if n.split() else n
    return k == n or k == first or (len(first) > 2 and re.fullmatch(re.escape(first) + r"( .*)?", k) is not None)
