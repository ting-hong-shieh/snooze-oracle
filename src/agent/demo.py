"""Build the static demo page: docs/demo/index.html (one file, data embedded as JSON).

    python -m src.agent demo

For each scenario: the plan (same code as the CLI), the TabPFN p_on_time for every minute
from 15 min before the first ring to 25 min after the wake time, and the backtest late rates
on that route for a fixed buffer (10 min commute, 30 min airport) and for the TabPFN alarm.
No model runs; nothing in the page calls a model.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.alarm.backtest import lookup_ride_quantile
from src.alarm.commute import GRID as COMMUTE_GRID, PREFIX as COMMUTE_PREFIX
from src.alarm.commute import CommuteScenario, lookup_median
from src.alarm.core import Scenario, _ride_table

from . import plan as P
from src.ride.retrieval import zone_borough

from . import zones

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("demo_template.html")
MAP = Path(__file__).with_name("demo_map.json")  # built by scripts/zones_to_map.py
OUT = ROOT / "docs" / "demo" / "index.html"
DAY = P.DEFAULT_DATE
PREP = 15.0

COMMUTE = [  # (from, to, kind)
    ("University Heights/Morris Heights", "Midtown Center", "sparse"),
    ("Cobble Hill", "Midtown North", "sparse"),
    ("TriBeCa/Civic Center", "Midtown Center", "dense"),
    ("Greenpoint", "Midtown East", "dense"),
]
DEADLINE = "09:00"
SHORT = {"University Heights/Morris Heights": "University Heights",
         "TriBeCa/Civic Center": "TriBeCa"}
AIRPORT_ZONE = {"JFK": 132, "LGA": 138}
PICKUP = [  # (flight, airport, from, label)
    ("DL709", "JFK", "South Ozone Park", "red-eye"),
    ("YX4451", "LGA", "Astoria", "short-haul"),
]


def _curve(s, wake: float) -> dict:
    m = np.arange(wake - P.FIRST_RING_MIN - 15, wake + 26, 1.0)
    return {"start": s.to_datetime(m[0]).strftime("%H:%M"),
            "p": [round(float(x), 4) for x in s.p_on_time(m)]}


def _late(path: Path, mask_cols: dict, strategies: tuple[str, str]) -> dict:
    r = pd.read_parquet(path)
    for c, v in mask_cols.items():
        r = r[r[c] == v]
    out = {}
    for s in strategies:
        x = r[r["strategy"] == s]
        out[s] = {"late": round(float(x["late"].mean()), 4),
                  "mornings": int(x["scenario"].nunique())}
    return out


def build() -> Path:
    scen = []
    lk_commute = _ride_table("lookup", COMMUTE_PREFIX, tuple(COMMUTE_GRID))
    lk_pickup = _ride_table("lookup")
    for origin, dest, kind in COMMUTE:
        p = P.plan_commute(origin, dest, DEADLINE, DAY, PREP)
        wake = p.pop("_wake_dt")
        s = CommuteScenario(p["from"]["zone"], p["to"]["zone"], DAY, P._minutes(DEADLINE),
                            "tabpfn", prep_min=PREP, prep_sd=P.PREP_SD, seed=0)
        wm = wake.hour * 60 + wake.minute
        rid = f"{p['from']['zone']}->{p['to']['zone']}"
        p["from"]["display"] = (f"{SHORT.get(p['from']['name'], p['from']['name'])} "
                                f"({zone_borough()[p['from']['zone']]})")
        bt = _late(ROOT / "data" / "commute" / "backtest_rows.parquet", {"route_id": rid},
                   ("gut_10", "tabpfn"))
        scen.append({
            "id": f"commute-{rid}", "scenario": "commute", "kind": kind,
            "title": f"{SHORT.get(p['from']['name'], p['from']['name'])} → {p['to']['name']}",
            "subtitle": f"Uber to a {DEADLINE} meeting",
            "deadline": DEADLINE, "plan": p, "curve": _curve(s, wm),
            # the median ride the fixed-buffer rule in the plan uses (same lookup, same minute)
            "map": {"from": p["from"]["zone"], "to": p["to"]["zone"],
                    "from_label": p["from"]["display"], "to_label": p["to"]["name"],
                    "ride_median_min": round(lookup_median(
                        lk_commute, (p["from"]["zone"], p["to"]["zone"], DAY),
                        P._minutes(DEADLINE) - 30))},
            "backtest": {"buffer_min": 10, "buffer": bt["gut_10"], "tabpfn": bt["tabpfn"],
                         "reality": ("real trips on the same route on any August weekday "
                                     "(sparse route)" if kind == "sparse" else
                                     "real trips on the same route and date")},
        })
    for flight, airport, origin, label in PICKUP:
        p = P.plan_pickup(flight, airport, origin, DAY, PREP)
        wake = p.pop("_wake_dt")
        f = p["flight"]
        fd = {"origin": f["origin"], "carrier": f["carrier"], "dest": airport,
              "crs_arr_ts": pd.Timestamp(f"{DAY} {f['scheduled_arrival']}")}
        s = Scenario(p["from"]["zone"], fd, "tabpfn", PREP, P.PREP_SD, P.EXIT_BUFFER, seed=0)
        wm = wake.hour * 60 + wake.minute
        bt = _late(ROOT / "data" / "alarm" / "backtest_rows.parquet",
                   {"pu_zone": p["from"]["zone"], "dest": airport}, ("gut_30", "tabpfn"))
        scen.append({
            "id": f"pickup-{flight}", "scenario": "pickup", "kind": label,
            "title": f"{f['carrier']}{f['number']} {f['origin']} → {airport}",
            "subtitle": f"pick someone up, leaving from {p['from']['name']}",
            "deadline": f["scheduled_arrival"], "plan": p, "curve": _curve(s, wm),
            "map": {"from": p["from"]["zone"], "to": AIRPORT_ZONE[airport],
                    "from_label": p["from"]["name"], "to_label": airport,
                    "ride_median_min": round(lookup_ride_quantile(
                        lk_pickup, (p["from"]["zone"], AIRPORT_ZONE[airport]), DAY,
                        s.crs + P.EXIT_BUFFER - 30, 0.5))},
            "backtest": {"buffer_min": 30, "buffer": bt["gut_30"], "tabpfn": bt["tabpfn"],
                         "reality": ("actual BTS arrival and real August trips on the same "
                                     "route within an hour, every flight landing 04:00-07:59")},
        })
    data = {"replay_date": str(DAY), "threshold": 0.9, "snooze_max_min": P.SNOOZE_MAX_MIN,
            "first_ring_min": P.FIRST_RING_MIN, "prep_min": PREP, "scenarios": scen,
            "map": json.loads(MAP.read_text())}
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", blob)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html) / 1024:.0f} KB, {len(scen)} scenarios)")
    return OUT
