"""Night-before precompute for the alarm: quantile tables the alarm reads.

Every model writes the same LEVELS (50 quantile levels, 0.01 ... 0.99), so Monte Carlo
sampling is identical across models: u ~ U(0.01, 0.99), value = interpolated quantile.

Ride tables, per model, one row per (pu_zone, do_zone, date, t_min), t_min on RIDE_GRID
(request minute of day, 03:00 ... 08:59):
    lookup    July 2026 empirical quantiles per (pu_zone, do_zone, req_hour, is_weekend),
              round-1 back-off (>= 10 rows, else route, else pickup zone, else global),
              all July rows in the product setting
    lgbm      LightGBM quantile (defaults, 4 threads), one model per level, all July rows
    tabpfn    TabPFN-3.5 on the r3 retrieval context of (route, hour), GPU only
Flight tables, per model, one row per flight (origin, carrier, dest, crs_arr_ts):
    lookup    empirical quantiles per origin x carrier (>= 20 rows), else origin, else all
    lgbm      LightGBM quantile on the "dest only" features, all training rows
    tabpfn    the round-2 "dest only" TabPFN grid (203 levels, full 28,672-row context),
              interpolated to LEVELS; nothing is recomputed
Quantile vectors are sorted per row (LightGBM levels can cross).

Usage (from the repo root):
    python -m src.alarm.tables ride --routes FILE --dates 2026-08-01:2026-08-31
    python -m src.alarm.tables flight
Writes data/alarm/ride_<model>.parquet and data/alarm/flight_<model>.parquet.
"""

from __future__ import annotations

import argparse
import fcntl
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "alarm"
LEVELS = np.round(np.linspace(0.01, 0.99, 50), 4)
LCOLS = [f"l{i:02d}" for i in range(len(LEVELS))]
RIDE_GRID = np.r_[np.arange(180, 531, 10), 539]  # request minute of day
MODELS = ("lookup", "lgbm", "tabpfn")


def _sorted(q: np.ndarray) -> np.ndarray:
    return np.sort(np.asarray(q, float), axis=1)


# ------------------------------------------------------------------ ride

def ride_queries(routes: list[tuple[int, int]], dates: list[pd.Timestamp],
                 grid: np.ndarray = RIDE_GRID) -> pd.DataFrame:
    rows = [(pu, do, d, t) for pu, do in routes for d in dates for t in grid]
    q = pd.DataFrame(rows, columns=["pu_zone", "do_zone", "date", "t_min"])
    q["req_min_of_day"] = q["t_min"]
    q["req_hour"] = q["t_min"] // 60
    q["dow"] = q["date"].dt.dayofweek
    q["is_weekend"] = (q["dow"] >= 5).astype(int)
    return q


def ride_lookup(q: pd.DataFrame, train: pd.DataFrame | None = None) -> np.ndarray:
    from src.ride.baselines import LOOKUP_LEVELS, LOOKUP_MIN_ROWS
    from src.ride.retrieval import load_train

    tr = load_train() if train is None else train
    out = np.full((len(q), len(LEVELS)), np.nan)
    todo = np.ones(len(q), bool)
    for keys in LOOKUP_LEVELS:
        if keys:
            g = tr.groupby(keys)["total_min"]
            n = g.size()
            t = g.quantile(list(LEVELS)).unstack()[n >= LOOKUP_MIN_ROWS]
            t.columns = LCOLS
            m = q.loc[todo, keys].merge(t.reset_index(), on=keys, how="left")
            vals = m[LCOLS].to_numpy()
        else:
            vals = np.tile(tr["total_min"].quantile(list(LEVELS)).to_numpy(), (todo.sum(), 1))
        hit = ~np.isnan(vals[:, 0])
        idx = np.flatnonzero(todo)[hit]
        out[idx] = vals[hit]
        todo[idx] = False
    return _sorted(out)


def ride_lgbm(q: pd.DataFrame, train: pd.DataFrame | None = None) -> np.ndarray:
    import lightgbm as lgb

    from src.ride.baselines import LGBM_THREADS, _lgbm_frame
    from src.ride.retrieval import load_train

    tr = load_train() if train is None else train
    Xtr, Xq = _lgbm_frame(tr), _lgbm_frame(q)
    out = np.zeros((len(q), len(LEVELS)))
    for j, a in enumerate(LEVELS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(a), n_jobs=LGBM_THREADS,
                              verbose=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, tr["total_min"])
            out[:, j] = m.predict(Xq)
    return _sorted(out)


def ride_tabpfn(q: pd.DataFrame, train: pd.DataFrame | None = None,
                hours: range | None = None) -> np.ndarray:
    """One fit per (route, hour) r3 context; takes the GPU lock per batch of contexts."""
    import torch

    from src.ride.common import FEATURES
    from src.ride.r2_tabpfn import LOCK, fit_context
    from src.ride.r3_context import context_rows

    out = np.zeros((len(q), len(LEVELS)))
    groups = list(q.groupby(["pu_zone", "do_zone", "req_hour"]).indices.items())
    i = 0
    with open(LOCK, "a") as lock:
        while i < len(groups):
            fcntl.flock(lock, fcntl.LOCK_EX)
            t0 = time.time()
            try:
                while i < len(groups) and time.time() - t0 < 120:
                    (pu, do, h), idx = groups[i]
                    ctx, _, _ = context_rows(pu, do, h, train, hours)
                    m = fit_context(ctx, "cuda")
                    res = m.predict(q.iloc[idx][FEATURES].to_numpy(np.float64),
                                    output_type="quantiles", quantiles=[float(a) for a in LEVELS])
                    out[idx] = np.column_stack(res)
                    del m
                    i += 1
                torch.cuda.empty_cache()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            print(f"ride tabpfn {i}/{len(groups)} contexts", flush=True)
            time.sleep(1)
    return _sorted(out)


def build_ride(routes, dates, models=MODELS, train: pd.DataFrame | None = None,
               grid: np.ndarray = RIDE_GRID, hours: range | None = None,
               prefix: str = "ride") -> None:
    """Defaults: the airport setting. The commute scenario passes its own training rows,
    request-time grid, hour window and file prefix."""
    OUT.mkdir(parents=True, exist_ok=True)
    q = ride_queries(routes, dates, grid)
    fn = {"lookup": lambda: ride_lookup(q, train), "lgbm": lambda: ride_lgbm(q, train),
          "tabpfn": lambda: ride_tabpfn(q, train, hours)}
    for m in models:
        t = time.time()
        vals = fn[m]()
        df = q[["pu_zone", "do_zone", "date", "t_min"]].copy()
        df[LCOLS] = vals
        df.to_parquet(OUT / f"{prefix}_{m}.parquet", index=False)
        print(f"ride {m}: {len(df):,} rows, {time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------ flight

FLIGHT_KEY = ["origin", "carrier", "dest", "crs_arr_ts"]


def flight_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, flown test rows) in the order the round-2 TabPFN grid was saved."""
    from src.flight.data import split

    train, test, _, _ = split(None)
    return train, test


def flight_lookup(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    out = np.full((len(test), len(LEVELS)), np.nan)
    todo = np.ones(len(test), bool)
    for keys in (["origin", "carrier"], ["origin"]):
        g = train.groupby(keys)["arr_delay"]
        t = g.quantile(list(LEVELS)).unstack()[g.size() >= 20]
        t.columns = LCOLS
        vals = test.loc[todo, keys].merge(t.reset_index(), on=keys, how="left")[LCOLS].to_numpy()
        hit = ~np.isnan(vals[:, 0])
        idx = np.flatnonzero(todo)[hit]
        out[idx] = vals[hit]
        todo[idx] = False
    out[todo] = train["arr_delay"].quantile(list(LEVELS)).to_numpy()
    return _sorted(out)


def flight_lgbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    from src.flight.data import Encoder
    from src.flight.evaluate2 import SEED, VARIANTS

    feats, cats = VARIANTS["dest only"]
    enc = Encoder(train, feats, cats)
    Xtr, Xte = enc.pandas_cat(train), enc.pandas_cat(test)
    out = np.zeros((len(test), len(LEVELS)))
    for j, a in enumerate(LEVELS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(a), num_threads=4,
                              random_state=SEED, verbose=-1)
        m.fit(Xtr, train["arr_delay"].to_numpy())
        out[:, j] = m.predict(Xte)
    return _sorted(out)


def flight_tabpfn(test: pd.DataFrame) -> np.ndarray:
    from src.flight.evaluate import GRID

    g = np.load(ROOT / "data" / "flight2" / "tabpfn_dest_only" / "grid.npy")
    assert g.shape == (len(test), len(GRID))
    return _sorted(np.stack([np.interp(LEVELS, GRID, row) for row in g]))


def build_flight(models=MODELS) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = flight_test()
    for m in models:
        vals = {"lookup": lambda: flight_lookup(train, test),
                "lgbm": lambda: flight_lgbm(train, test),
                "tabpfn": lambda: flight_tabpfn(test)}[m]()
        df = test[FLIGHT_KEY].copy()
        df[LCOLS] = vals
        df.to_parquet(OUT / f"flight_{m}.parquet", index=False)
        print(f"flight {m}: {len(df):,} rows", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["ride", "flight"])
    ap.add_argument("--routes", help="CSV with pu_zone,do_zone")
    ap.add_argument("--dates", default="2026-08-01:2026-08-31")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    models = a.models.split(",")
    if a.what == "flight":
        build_flight(models)
    else:
        r = pd.read_csv(a.routes)
        d0, d1 = a.dates.split(":")
        build_ride(list(zip(r.pu_zone, r.do_zone)), list(pd.date_range(d0, d1)), models)
