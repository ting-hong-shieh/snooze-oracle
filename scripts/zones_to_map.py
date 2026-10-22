"""NYC TLC taxi zones -> the compact map geometry the demo page embeds.

    ~/.venvs/snooze-media/bin/python -I scripts/zones_to_map.py \
        data/raw/taxi_zones/taxi_zones/taxi_zones.shp src/agent/demo_map.json

Needs pyshp (not in the project venv; the output is committed, so `python -m src.agent demo`
does not need it). Input is EPSG:2263 (feet). Output: zones clipped to the part of the city the
demo routes cross, Douglas-Peucker simplified, as SVG path data in 25-ft units with y pointing
down, plus borough outlines (zone edges that do not touch another zone of the same borough),
and zone centroids (labels come from the plan). No streets: the page draws a straight-line sketch, not a route.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import shapefile

UNIT = 25.0                                        # feet per map unit
X0, X1, Y0, Y1 = 958000, 1066000, 150000, 268000   # clip window (ft): Bronx to JFK, W of the Hudson
TOL_ZONE, TOL_BORO = 90.0, 110.0                   # simplification tolerance (ft)
MIN_LINE = 700.0                                   # shortest open outline piece kept (ft)
SKIP = {264, 265}                                  # "Unknown" / "Outside of NYC"


def dp(pts, tol):
    """Douglas-Peucker, iterative; keeps both ends."""
    if len(pts) < 3:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (ax, ay), (bx, by) = pts[a], pts[b]
        dx, dy = bx - ax, by - ay
        n = (dx * dx + dy * dy) ** 0.5
        best, bi = -1.0, -1
        for i in range(a + 1, b):
            px, py = pts[i]
            d = abs(dy * (px - ax) - dx * (py - ay)) / n if n else ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            if d > best:
                best, bi = d, i
        if best > tol:
            keep[bi] = True
            stack += [(a, bi), (bi, b)]
    return [p for p, k in zip(pts, keep) if k]


def to_u(p):
    return (round((p[0] - X0) / UNIT), round((Y1 - p[1]) / UNIT))


def path(lines, closed):
    """SVG path data with relative moves; drops repeated points after rounding."""
    out = []
    for pts in lines:
        q = [to_u(p) for p in pts]
        q = [p for i, p in enumerate(q) if i == 0 or p != q[i - 1]]
        if len(q) < (3 if closed else 2):
            continue
        s = f"M{q[0][0]} {q[0][1]}"
        for (x0, y0), (x1, y1) in zip(q, q[1:]):
            s += f"l{x1 - x0} {y1 - y0}"
        out.append(s + ("z" if closed else ""))
    return "".join(out).replace(" -", "-")


def centroid(rings):
    A = cx = cy = 0.0
    for ring in rings:
        for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
            c = x0 * y1 - x1 * y0
            A += c; cx += (x0 + x1) * c; cy += (y0 + y1) * c
    return (cx / (3 * A), cy / (3 * A))


def chain(edges):
    """Join undirected segments into polylines."""
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    used, lines = set(), []
    for a, b in edges:
        if (a, b) in used or (b, a) in used:
            continue
        used.add((a, b))
        line = [a, b]
        for end in (1, 0):
            while True:
                tip = line[-1] if end else line[0]
                nxt = [n for n in adj[tip] if (tip, n) not in used and (n, tip) not in used]
                if len(adj[tip]) != 2 or not nxt:
                    break
                n = nxt[0]
                used.add((tip, n))
                if end: line.append(n)
                else: line.insert(0, n)
        lines.append(line)
    return lines


def main(shp: str, out: str) -> None:
    r = shapefile.Reader(shp)
    fields = [f[0] for f in r.fields[1:]]
    zones, boro_edges = [], defaultdict(lambda: defaultdict(int))
    for sr in r.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        zid = int(rec["LocationID"])
        bx0, by0, bx1, by1 = sr.shape.bbox
        if zid in SKIP or bx1 < X0 or bx0 > X1 or by1 < Y0 or by0 > Y1:
            continue
        pts, parts = sr.shape.points, list(sr.shape.parts) + [len(sr.shape.points)]
        rings = [[(round(x), round(y)) for x, y in pts[a:b]] for a, b in zip(parts[:-1], parts[1:])]
        for ring in rings:
            for a, b in zip(ring, ring[1:]):
                if a != b:
                    boro_edges[rec["borough"]][(min(a, b), max(a, b))] += 1
        simp = [dp(ring, TOL_ZONE) for ring in rings]
        c = centroid(rings)
        zones.append({"id": zid, "c": list(to_u(c)), "d": path(simp, True)})
    boroughs = {}
    for b, edges in boro_edges.items():
        # an edge drawn by two zones of the same borough is interior; the rest is the outline
        outline = [e for e, n in edges.items() if n == 1]
        lines = [dp(l, TOL_BORO) for l in chain(outline)]
        # drop stray slivers where neighbouring zones do not share vertices exactly
        lines = [l for l in lines if l[0] == l[-1] or sum(
            ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(l, l[1:])) > MIN_LINE]
        boroughs[b] = path(lines, False)
    data = {"unit_ft": UNIT, "size": [round((X1 - X0) / UNIT), round((Y1 - Y0) / UNIT)],
            "zones": zones, "boroughs": boroughs}
    s = json.dumps(data, separators=(",", ":"))
    open(out, "w").write(s + "\n")
    print(f"{len(zones)} zones, {len(s) / 1024:.1f} KB "
          f"(zones {sum(len(z['d']) for z in zones) / 1024:.1f} KB, "
          f"boroughs {sum(len(v) for v in boroughs.values()) / 1024:.1f} KB)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
