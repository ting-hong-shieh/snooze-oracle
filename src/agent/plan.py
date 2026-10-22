"""Night-before plan from precomputed tables only (no model runs, no GPU).

The data ends in August 2026, so the planned morning is a replay of an August date. The
.ics file puts the same clock times on --alarm-date (default: tomorrow on this computer)
so the alarms can actually ring.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.alarm.backtest import lookup_ride_quantile
from src.alarm.commute import ROUTES as COMMUTE_ROUTES
from src.alarm.commute import CommuteScenario, lookup_median, weekdays
from src.alarm.core import THRESHOLD, Scenario, _ride_table
from src.alarm.commute import GRID as COMMUTE_GRID, PREFIX as COMMUTE_PREFIX

from . import ics, index, zones

ROOT = Path(__file__).resolve().parents[2]
AIRPORT_ROUTES = ROOT / "data" / "ride3" / "backtest_routes.csv"
OUT = ROOT / "out"
DEFAULT_DATE = date(2026, 8, 12)   # a Wednesday in the test month
PREP_SD = 5.0
EXIT_BUFFER = 20.0
FIRST_RING_MIN = 30
SNOOZE_MAX_MIN = 10  # Google Clock's default snooze; the backtest (src/alarm/backtest.py) used 9
GUT_B = {"commute": 10, "pickup": 30}
REPLAY_NOTE = ("Replay: the data ends in August 2026, so this plans the morning of {d} "
               "using what was known the night before. Alarms in the .ics file use the same "
               "clock times on {a}.")


class PlanError(Exception):
    pass


def _hhmm(t: datetime) -> str:
    return t.strftime("%H:%M")


def _minutes(hhmm: str) -> int:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", hhmm.strip())
    if not m or int(m[1]) > 23 or int(m[2]) > 59:
        raise PlanError(f"time must be HH:MM, got '{hhmm}'")
    return int(m[1]) * 60 + int(m[2])


def _supported(path: Path, do_zone: int | None = None) -> dict[int, str]:
    r = pd.read_csv(path)
    if do_zone is not None:
        r = r[r["do_zone"] == do_zone]
    names = zones.zones()
    return {int(z): names[int(z)] for z in r["pu_zone"]}


def _pick(query: str, allowed: dict[int, str], what: str) -> int:
    """Match against every TLC zone (asking when ambiguous), then require a precomputed one."""
    zid = zones.choose(query)
    if zid not in allowed:
        raise PlanError(f"{zones.zones()[zid]} is not a precomputed {what}. Supported: "
                        + ", ".join(sorted(allowed.values())))
    return zid


def _result(s_tab, s_req, s_lk, gut_wake: float, b: int) -> dict:
    wake, ok = s_tab.latest_safe_wake()
    if not ok:
        raise PlanError("no wake time in the search window reaches 90%")
    req, _ = s_req.latest_safe_wake()
    lk_wake, _ = s_lk.latest_safe_wake()
    t = s_tab.to_datetime
    return {
        "wake_time": _hhmm(t(wake)),
        "first_ring": _hhmm(t(wake - FIRST_RING_MIN)),
        "latest_request_time": _hhmm(t(req)),
        "p_on_time_at_wake": round(float(s_tab.p_on_time(wake)[0]), 3),
        "alternatives": {
            "lookup": {"wake_time": _hhmm(t(lk_wake)),
                       "p_on_time_tabpfn": round(float(s_tab.p_on_time(lk_wake)[0]), 3)},
            "gut_rule": {"buffer_min": b, "wake_time": _hhmm(t(gut_wake)),
                         "p_on_time_tabpfn": round(float(s_tab.p_on_time(gut_wake)[0]), 3)},
        },
        "_wake_dt": t(wake),
    }


def plan_commute(origin: str, dest: str, deadline: str, day: date, prep_min: float) -> dict:
    routes = pd.read_csv(COMMUTE_ROUTES)
    do = _pick(dest, {int(z): zones.zones()[int(z)] for z in routes["do_zone"].unique()},
               "commute destination")
    pu = _pick(origin, _supported(COMMUTE_ROUTES, do), f"commute origin for {zones.zones()[do]}")
    if not ((routes.pu_zone == pu) & (routes.do_zone == do)).any():
        raise PlanError("route not precomputed. Supported commute routes:\n" + "\n".join(
            f"  {zones.zones()[a]} -> {zones.zones()[b]}" for a, b in zip(routes.pu_zone,
                                                                          routes.do_zone)))
    if pd.Timestamp(day) not in weekdays():
        raise PlanError("commute replay dates are August 2026 weekdays")
    meet = _minutes(deadline)
    mk = lambda m, p, sd: CommuteScenario(pu, do, day, meet, m, prep_min=p, prep_sd=sd, seed=0)
    lk = _ride_table("lookup", COMMUTE_PREFIX, tuple(COMMUTE_GRID))
    b = GUT_B["commute"]
    gut = meet - prep_min - lookup_median(lk, (pu, do, day), meet - 30) - b
    res = _result(mk("tabpfn", prep_min, PREP_SD), mk("tabpfn", 0.0, 0.0),
                  mk("lookup", prep_min, PREP_SD), gut, b)
    return {"scenario": "commute", "from": {"zone": pu, "name": zones.zones()[pu]},
            "to": {"zone": do, "name": zones.zones()[do]}, "deadline": deadline, **res}


def _parse_flight(s: str) -> tuple[str, int]:
    m = re.fullmatch(r"\s*([A-Za-z0-9]{2})\s*(\d{1,4})\s*", s)
    if not m:
        raise PlanError(f"flight must look like DL1234, got '{s}'")
    return m[1].upper(), int(m[2])


def plan_pickup(flight: str, airport: str, origin: str, day: date, prep_min: float) -> dict:
    carrier, num = _parse_flight(flight)
    f = index.load()
    arr = pd.to_datetime(f["crs_arr_ts"])
    hit = f[(f.carrier == carrier) & (f.flight_num == num) & (f.dest == airport)
            & (arr.dt.date == day)]
    if hit.empty:
        raise PlanError(f"no precomputed {carrier}{num} into {airport} landing on {day} "
                        f"(03:00-08:59). List them with: python -m src.agent flights "
                        f"--airport {airport} --date {day}")
    fl = hit.iloc[0]
    do = {"JFK": 132, "LGA": 138}[airport]
    pu = _pick(origin, _supported(AIRPORT_ROUTES, do), f"pickup zone for {airport}")
    fd = {"origin": fl.origin, "carrier": fl.carrier, "dest": fl.dest,
          "crs_arr_ts": pd.Timestamp(fl.crs_arr_ts)}
    mk = lambda m, p, sd: Scenario(pu, fd, m, p, sd, EXIT_BUFFER, seed=0)
    s_tab = mk("tabpfn", prep_min, PREP_SD)
    b = GUT_B["pickup"]
    lk = _ride_table("lookup")
    gut = (s_tab.crs + EXIT_BUFFER - prep_min
           - lookup_ride_quantile(lk, (pu, do), day, s_tab.crs + EXIT_BUFFER - 30, 0.5) - b)
    res = _result(s_tab, mk("tabpfn", 0.0, 0.0), mk("lookup", prep_min, PREP_SD), gut, b)
    return {"scenario": "pickup",
            "flight": {"carrier": fl.carrier, "number": int(fl.flight_num), "origin": fl.origin,
                       "airport": airport, "scheduled_arrival": _hhmm(fd["crs_arr_ts"]),
                       "p_cancel": round(float(fl.p_cancel), 3)},
            "from": {"zone": pu, "name": zones.zones()[pu]},
            "exit_buffer_min": EXIT_BUFFER, **res}


def write(plan: dict, day: date, alarm_day: date, prep_min: float) -> dict:
    wake_dt = plan.pop("_wake_dt")
    plan = {"replay_date": str(day), "alarm_date": str(alarm_day),
            "note": REPLAY_NOTE.format(d=day, a=alarm_day), "model": "TabPFN-3.5",
            "threshold": THRESHOLD, "prep_min": prep_min, "prep_sd": PREP_SD,
            "snooze_max_min": SNOOZE_MAX_MIN, **plan}
    OUT.mkdir(exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    ring = datetime.combine(alarm_day, wake_dt.time())
    where = plan["to"]["name"] if plan["scenario"] == "commute" else plan["flight"]["airport"]
    (OUT / "alarms.ics").write_text(ics.make(
        ring, f"Wake up for {where} (Snooze Oracle)",
        f"Latest safe wake {plan['wake_time']}, request the ride by "
        f"{plan['latest_request_time']}. {plan['note']}", FIRST_RING_MIN))
    return plan


def default_alarm_day() -> date:
    return date.today() + timedelta(days=1)
