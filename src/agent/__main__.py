"""Snooze Oracle planner CLI.

    python -m src.agent index      # once: flight index for pickups
    python -m src.agent plan --scenario commute --from "Cobble Hill" --to "Midtown North" \\
        --deadline 09:00 [--date 2026-08-12] [--prep-min 15] [--alarm-date 2026-10-24]
    python -m src.agent plan --scenario pickup --flight DL1234 --airport JFK \\
        --from "South Ozone Park" [--date ...]
    python -m src.agent flights --airport JFK --date 2026-08-12

Writes out/plan.json and out/alarms.ics. Uses precomputed tables only.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from . import zones


def _date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    from .plan import DEFAULT_DATE

    ap = argparse.ArgumentParser(prog="python -m src.agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build the flight index once")
    sub.add_parser("demo", help="build the static demo page docs/demo/index.html")
    p = sub.add_parser("plan", help="plan a wake time")
    p.add_argument("--scenario", choices=["commute", "pickup"], required=True)
    p.add_argument("--from", dest="origin", required=True, help="pickup zone name")
    p.add_argument("--to", help="commute: destination zone name")
    p.add_argument("--deadline", help="commute: meeting time HH:MM")
    p.add_argument("--flight", help="pickup: carrier + number, e.g. DL1234")
    p.add_argument("--airport", choices=["JFK", "LGA"], help="pickup: arrival airport")
    p.add_argument("--prep-min", type=float, default=15.0)
    p.add_argument("--date", type=_date, default=DEFAULT_DATE,
                   help=f"replay date in August 2026 (default {DEFAULT_DATE})")
    p.add_argument("--alarm-date", type=_date, default=None,
                   help="date to put the .ics alarms on (default: tomorrow)")
    fl = sub.add_parser("flights", help="list precomputed flights")
    fl.add_argument("--airport", choices=["JFK", "LGA"], required=True)
    fl.add_argument("--date", type=_date, default=DEFAULT_DATE)
    a = ap.parse_args(argv)

    if a.cmd == "index":
        from .index import build
        build()
        return 0
    if a.cmd == "demo":
        from .demo import build as build_demo
        build_demo()
        return 0
    if a.cmd == "flights":
        import pandas as pd

        from .index import load
        f = load()
        f = f[(f.dest == a.airport) & (pd.to_datetime(f.crs_arr_ts).dt.date == a.date)]
        for r in f.sort_values("crs_arr_ts").itertuples():
            print(f"{r.carrier}{r.flight_num:<5} {r.origin} -> {r.dest} "
                  f"{pd.Timestamp(r.crs_arr_ts):%H:%M}")
        print(f"{len(f)} flights")
        return 0

    from .plan import PlanError, default_alarm_day, plan_commute, plan_pickup, write
    try:
        if a.scenario == "commute":
            if not (a.to and a.deadline):
                raise PlanError("commute needs --to and --deadline")
            plan = plan_commute(a.origin, a.to, a.deadline, a.date, a.prep_min)
        else:
            if not (a.flight and a.airport):
                raise PlanError("pickup needs --flight and --airport")
            plan = plan_pickup(a.flight, a.airport, a.origin, a.date, a.prep_min)
    except zones.Ambiguous as e:
        print(f"'{e.query}' matches several zones; pass one of these names:", file=sys.stderr)
        for zid, name in e.options:
            print(f"  {name} (zone {zid})", file=sys.stderr)
        return 2
    except (PlanError, zones.NoMatch) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    plan = write(plan, a.date, a.alarm_date or default_alarm_day(), a.prep_min)
    print(plan["note"])
    print(f"Wake at {plan['wake_time']} (first ring {plan['first_ring']}); request the ride "
          f"by {plan['latest_request_time']}; P(on time) at {plan['wake_time']}: "
          f"{plan['p_on_time_at_wake']:.0%}.")
    g = plan["alternatives"]["gut_rule"]
    print(f"Gut rule ({g['buffer_min']} min buffer) would wake you at {g['wake_time']}: "
          f"P(on time) {g['p_on_time_tabpfn']:.0%}. Lookup table: "
          f"{plan['alternatives']['lookup']['wake_time']}.")
    print("Wrote out/plan.json and out/alarms.ics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
