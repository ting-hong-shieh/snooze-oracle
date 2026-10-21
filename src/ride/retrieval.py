"""Round 2: the product-aligned setting and the TabPFN retrieval context.

Product setting: rides to JFK (132) or LaGuardia (138) requested 03:00-08:59.
Train = July 2026 trips in that setting, test = August 2026 trips in that setting.

Retrieval (one context per query key, shared by every query with that key):
    route level    trips with the same pu_zone -> do_zone whose req_hour is within
                   +-HOUR_RADIUS of the query's req_hour;
    borough level  if the route level has fewer than MIN_ROUTE_ROWS rows: the same
                   pu_borough -> do_zone, same hour band;
    cap            at most MAX_CONTEXT_ROWS rows, a uniform random sample without
                   replacement, seeded by the key (so a key always gets the same rows).
The hour band is taken from the window-restricted July rows, so it is truncated (not
wrapped) at the edges: a 03:xx query sees hours 3-5, an 08:xx query hours 6-8.
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT2 = ROOT / "data" / "ride2"
TRAIN_FILE = PROCESSED / "uber_trips_2026-07.parquet"
TEST_FILE = PROCESSED / "uber_trips_2026-08.parquet"
ZONE_LOOKUP = ROOT / "data" / "raw" / "taxi_zone_lookup.csv"

DEST_ZONES = (132, 138)          # JFK, LaGuardia
WINDOW_MIN = (180, 539)          # request 03:00-08:59, req_min_of_day inclusive
WINDOW_HOURS = range(3, 9)
HOUR_RADIUS = 2
MIN_ROUTE_ROWS = 500
MAX_CONTEXT_ROWS = 10_000

COLS = ("request_ts, pu_zone, do_zone, pu_borough, req_hour, req_min_of_day, dow, "
        "CAST(is_weekend AS INTEGER) AS is_weekend, total_min")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'; SET threads=4;")
    return con


def window_sql(path: Path, month_start: str, month_end: str) -> str:
    """Trips in the product setting, strict month split on the request time."""
    return f"""
        SELECT {COLS} FROM read_parquet('{path}')
        WHERE do_zone IN {DEST_ZONES}
          AND req_min_of_day BETWEEN {WINDOW_MIN[0]} AND {WINDOW_MIN[1]}
          AND request_ts >= TIMESTAMP '{month_start}' AND request_ts < TIMESTAMP '{month_end}'
    """


@lru_cache(maxsize=1)
def load_train() -> pd.DataFrame:
    """All July trips in the product setting (~225k rows, ~20 MB)."""
    path = OUT2 / "train.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return connect().execute(window_sql(TRAIN_FILE, "2026-07-01", "2026-08-01")
                             + " ORDER BY request_ts, pu_zone, do_zone").fetchdf()


@lru_cache(maxsize=1)
def zone_borough() -> dict[int, str]:
    z = pd.read_csv(ZONE_LOOKUP)
    return dict(zip(z["LocationID"].astype(int), z["Borough"].astype(str)))


def hour_band(h: int) -> tuple[int, int]:
    return max(h - HOUR_RADIUS, WINDOW_HOURS.start), min(h + HOUR_RADIUS, WINDOW_HOURS.stop - 1)


@lru_cache(maxsize=1)
def _counts():
    tr = load_train()
    route = tr.groupby(["pu_zone", "do_zone", "req_hour"]).size()
    boro = tr.groupby(["pu_borough", "do_zone", "req_hour"]).size()
    return route.to_dict(), boro.to_dict()


def context_key(pu_zone: int, do_zone: int, req_hour: int,
                pu_borough: str | None = None) -> tuple:
    """('route', pu_zone, do_zone, h) or ('borough', pu_borough, do_zone, h)."""
    route, _ = _counts()
    lo, hi = hour_band(req_hour)
    n_route = sum(route.get((pu_zone, do_zone, h), 0) for h in range(lo, hi + 1))
    if n_route >= MIN_ROUTE_ROWS:
        return ("route", int(pu_zone), int(do_zone), int(req_hour))
    if pu_borough is None:
        pu_borough = zone_borough()[int(pu_zone)]
    return ("borough", str(pu_borough), int(do_zone), int(req_hour))


def key_name(key: tuple) -> str:
    return "_".join(str(k).replace(" ", "-") for k in key)


def context_rows(key: tuple) -> tuple[pd.DataFrame, int]:
    """(context rows, rows matched before the cap)."""
    tr = load_train()
    level, place, do_zone, h = key
    lo, hi = hour_band(h)
    col = "pu_zone" if level == "route" else "pu_borough"
    m = (tr[col] == place) & (tr["do_zone"] == do_zone) & tr["req_hour"].between(lo, hi)
    ctx = tr[m]
    n_match = len(ctx)
    if n_match > MAX_CONTEXT_ROWS:
        rng = np.random.default_rng(zlib.crc32(key_name(key).encode()))
        ctx = ctx.iloc[np.sort(rng.choice(n_match, MAX_CONTEXT_ROWS, replace=False))]
    return ctx, n_match
