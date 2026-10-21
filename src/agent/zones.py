"""Fuzzy matching of a typed place name to a TLC taxi zone."""

from __future__ import annotations

import difflib
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOOKUP = ROOT / "data" / "raw" / "taxi_zone_lookup.csv"


class Ambiguous(Exception):
    def __init__(self, query: str, options: list[tuple[int, str]]):
        self.query, self.options = query, options
        super().__init__(f"'{query}' matches several zones")


class NoMatch(Exception):
    pass


@lru_cache(maxsize=1)
def zones() -> dict[int, str]:
    z = pd.read_csv(LOOKUP)
    z = z[~z["LocationID"].isin([264, 265])]
    return dict(zip(z["LocationID"].astype(int), z["Zone"].astype(str)))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def match(query: str, allowed: dict[int, str] | None = None) -> int:
    """Zone id for `query`. Exact (case-insensitive) or a unique substring match wins;
    otherwise difflib close matches. Several candidates raise Ambiguous."""
    cand = allowed if allowed is not None else zones()
    q = _norm(query)
    if q.isdigit() and int(q) in cand:
        return int(q)
    names = {zid: _norm(n) for zid, n in cand.items()}
    exact = [z for z, n in names.items() if n == q]
    if len(exact) == 1:
        return exact[0]
    # whole words of the query inside the zone name ("midtown center", "cobble")
    sub = [z for z, n in names.items() if q and all(w in n.split() or w in n for w in q.split())]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise Ambiguous(query, [(z, cand[z]) for z in sorted(sub)])
    close = difflib.get_close_matches(q, list(names.values()), n=5, cutoff=0.6)
    ids = [z for c in close for z, n in names.items() if n == c]
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:  # a clear typo winner: best ratio high and well ahead of the next
        r = [difflib.SequenceMatcher(None, q, names[z]).ratio() for z in ids]
        if r[0] >= 0.85 and r[0] - r[1] >= 0.08:
            return ids[0]
    if len(ids) > 1:
        raise Ambiguous(query, [(z, cand[z]) for z in ids])
    raise NoMatch(f"no zone matches '{query}'")


def choose(query: str, allowed: dict[int, str] | None = None) -> int:
    """match(), asking the user on a terminal when several zones fit."""
    try:
        return match(query, allowed)
    except Ambiguous as e:
        if not sys.stdin.isatty():
            raise
        print(f"'{query}' matches several zones:")
        for i, (zid, name) in enumerate(e.options, 1):
            print(f"  {i}. {name} (zone {zid})")
        while True:
            a = input(f"Choose 1-{len(e.options)}: ").strip()
            if a.isdigit() and 1 <= int(a) <= len(e.options):
                return e.options[int(a) - 1][0]
