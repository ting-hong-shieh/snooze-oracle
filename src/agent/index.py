"""One-time index for the agent: August 2026 flights with a precomputed TabPFN delay table.

    python -m src.agent index

Writes data/agent/flights.parquet: origin, carrier, flight_num, dest, crs_arr_ts,
flight_date, distance, p_cancel. Only flights that have rows in data/alarm/flight_*.parquet
(flown flights into JFK/LGA scheduled 03:00-08:59, FlightDate in August 2026) are listed, so
the planner never needs a model at run time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "agent"
FLIGHTS = OUT / "flights.parquet"


def build() -> pd.DataFrame:
    from src.alarm.tables import flight_test
    from src.flight.model import p_cancel

    _, test = flight_test()
    tab = pd.read_parquet(ROOT / "data" / "alarm" / "flight_tabpfn.parquet")
    assert len(tab) == len(test)
    f = test[["origin", "carrier", "flight_num", "dest", "crs_arr_ts", "flight_date",
              "distance"]].copy()
    f["p_cancel"] = [p_cancel(o, c, t, d) for o, c, t, d in
                     zip(f.origin, f.carrier, f.crs_arr_ts, f.dest)]
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(FLIGHTS, index=False)
    print(f"wrote {FLIGHTS.relative_to(ROOT)} ({len(f):,} flights)")
    return f


def load() -> pd.DataFrame:
    if not FLIGHTS.exists():
        raise SystemExit("flight index missing: run `python -m src.agent index` once")
    return pd.read_parquet(FLIGHTS)
