"""Regime features known the night before: airport-level delays two days before arrival.

For a flight scheduled to arrive at JFK/LGA on NYC-local date A (the date of `crs_arr_ts`):

    origin_prev_dep_delay = mean DepDelay of ALL BTS departures from the flight's origin
                            with FlightDate = A - 2 (origin-local departure date)
    dest_prev_arr_delay   = mean ArrDelay of ALL BTS arrivals at the flight's dest (JFK or
                            LGA) scheduled to arrive on NYC-local date A - 2

Why A - 2 and not A - 1: predictions are precomputed the evening of A - 1 (SPEC: "the night
before"). The previous calendar day of that precompute is A - 2, the last day whose flights
have all departed and landed by then. Day A - 1 is still running at precompute time, and
BTS FlightDate A - 1 also contains the red-eyes that land on the morning of A, i.e. the
target flights themselves (a direct leak). Anchoring on the scheduled arrival date is
required because `crs_arr_ts` is the only date the interface receives.

Means use flights with a non-null delay: cancelled flights (no departure, no arrival) and,
for the arrival mean, diverted flights (no arrival at dest) drop out. Diverted flights do
have a DepDelay and stay in the origin mean. No clipping: early departures (negative delay)
count as negative. An airport with no qualifying flights on A - 2 gives NaN (both models
handle missing values natively); the count is stored next to the mean.

    python -m src.flight.regime     # writes data/flight2/airport_day.parquet
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .data import ROOT

OUT2 = ROOT / "data" / "flight2"
RAW = ROOT / "data" / "raw" / "bts"
AIRPORT_DAY = OUT2 / "airport_day.parquet"
BTS = "https://transtats.bts.gov/PREZIP"
# 2025-07 is needed so that training arrivals on 2025-08-01/02 have their A - 2 day.
MONTHS = ["2025-07"] + [f"2025-{m:02d}" for m in range(8, 13)] + [f"2026-{m:02d}" for m in range(1, 9)]
LAG_DAYS = 2
REGIME_FEATURES = ["origin_prev_dep_delay", "dest_prev_arr_delay"]


def ensure_csv(month: str) -> Path:
    csv = RAW / f"ontime_{month}.csv"
    if csv.exists():
        return csv
    sys.path.insert(0, str(ROOT / "scripts"))
    from prep_trips import download  # shared helper; scripts/ is not edited

    year, mon = month.split("-")
    name = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{int(mon)}"
    z = download(f"{BTS}/{name}.zip", RAW / f"{name}.zip")
    with zipfile.ZipFile(z) as zf:
        member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
        tmp = csv.with_suffix(".csv.part")
        with zf.open(member) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        tmp.rename(csv)
    return csv


def build() -> pd.DataFrame:
    files = [str(ensure_csv(m)) for m in MONTHS]
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    df = con.execute(f"""
        WITH f AS (
          SELECT CAST(FlightDate AS DATE) AS fd, Origin, Dest,
                 CAST(CRSDepTime AS INTEGER) AS dep, CAST(CRSArrTime AS INTEGER) AS arr,
                 TRY_CAST(DepDelay AS DOUBLE) AS dep_delay,
                 TRY_CAST(ArrDelay AS DOUBLE) AS arr_delay
          FROM read_csv({files!r}, header = true, all_varchar = true, union_by_name = true)
        )
        SELECT 'origin_dep' AS kind, Origin AS airport, fd AS day,
               avg(dep_delay) AS mean_delay, count(dep_delay) AS n_flights, count(*) AS n_sched
        FROM f GROUP BY ALL
        UNION ALL
        SELECT 'dest_arr', Dest,
               -- scheduled arrival date, NYC local (same red-eye rule as prep_flights.py)
               CAST(fd + CAST(arr < dep OR arr = 2400 AS INTEGER) AS DATE),
               avg(arr_delay), count(arr_delay), count(*)
        FROM f WHERE Dest IN ('JFK', 'LGA') GROUP BY ALL
    """).df()
    df["day"] = pd.to_datetime(df["day"]).dt.date
    OUT2.mkdir(parents=True, exist_ok=True)
    df.to_parquet(AIRPORT_DAY)
    return df


def _table() -> pd.DataFrame:
    if not AIRPORT_DAY.exists():
        build()
    return pd.read_parquet(AIRPORT_DAY)


def add_regime(df: pd.DataFrame, table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add REGIME_FEATURES to rows with `origin`, `dest`, `crs_arr_ts`. Days with no
    qualifying flights (or outside the table) give NaN."""
    t = _table() if table is None else table
    t = t[t["n_flights"] > 0]
    key = pd.to_datetime(df["crs_arr_ts"]).dt.date - pd.Timedelta(days=LAG_DAYS)
    o = t[t["kind"] == "origin_dep"].set_index(["airport", "day"])["mean_delay"]
    d = t[t["kind"] == "dest_arr"].set_index(["airport", "day"])["mean_delay"]
    out = df.copy()
    out["origin_prev_dep_delay"] = o.reindex(list(zip(df["origin"], key))).to_numpy()
    out["dest_prev_arr_delay"] = d.reindex(list(zip(df["dest"], key))).to_numpy()
    return out


if __name__ == "__main__":
    t = build()
    print(t.groupby("kind").agg(rows=("day", "size"), first=("day", "min"), last=("day", "max")))
    a = t[t.kind == "dest_arr"].pivot(index="day", columns="airport", values="mean_delay").sort_index()
    for ap in a:
        s = a[ap]
        print(ap, "daily mean ArrDelay autocorr lag1 %.3f lag2 %.3f" % (s.autocorr(1), s.autocorr(2)))
