"""Runtime interface (see SPEC.md "Interfaces"):

    flight_arrival_samples(origin, carrier, crs_arr_ts, n=2000) -> np.ndarray
    p_cancel(origin, carrier, crs_arr_ts) -> float

flight_arrival_samples uses TabPFN-3.5 fitted on the same context rows as the 2026-08
evaluation (all flown 2025-08 ... 2026-07 arrivals scheduled 03:00-08:59; nothing is pickled,
so no weights are written to disk). It takes TabPFN's predictive distribution on a fine
quantile grid and draws n samples by inverse-CDF interpolation on that grid.

One TabPFN predict call costs minutes on the shared GPU whether it is for 1 row or 500 (the
~29k-row context dominates), so call `precompute([...])` for all flights of interest in one
batch the night before; grids are stored in data/flight/grid_cache.parquet and
flight_arrival_samples reads them from there. A flight that is not in the cache is predicted
on the spot.

The GPU is shared on the dev box: run any process that may call TabPFN under
`flock /tmp/snooze-oracle-gpu.lock ...`, like the evaluation stage. Set
SNOOZE_FLIGHT_DEVICE=cpu to run on CPU instead.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .cancel import CancelModel
from .data import OUT, SEED, TRAIN_MONTHS, Encoder, features_from_query, load_months, split
from .evaluate import GRID

CACHE = OUT / "grid_cache.parquet"
_mem: dict[tuple, np.ndarray] = {}


@functools.lru_cache(maxsize=1)
def _train():
    train, _, _, _ = split(None)
    return train


@functools.lru_cache(maxsize=1)
def _regressor():
    from tabpfn import TabPFNRegressor

    train = _train()
    enc = Encoder(train)
    device = os.environ.get("SNOOZE_FLIGHT_DEVICE", "cuda")
    m = TabPFNRegressor(device=device, categorical_features_indices=[0, 1], random_state=SEED)
    m.fit(enc.numeric(train), train["arr_delay"].to_numpy())
    return enc, m


def _distance(origin: str, carrier: str) -> float:
    """The interface has no destination, so use the most common training distance for the
    (origin, carrier) cell, else for the origin, else the overall median."""
    t = _train()
    for d in (t[(t.origin == origin) & (t.carrier == carrier)], t[t.origin == origin]):
        if len(d):
            return float(d["distance"].mode().iloc[0])
    return float(t["distance"].median())


def _key(origin: str, carrier: str, crs_arr_ts) -> tuple:
    return (origin, carrier, pd.Timestamp(crs_arr_ts).floor("min"))


def _load_cache() -> None:
    if not _mem and CACHE.exists():
        for r in pd.read_parquet(CACHE).itertuples(index=False):
            _mem[(r.origin, r.carrier, pd.Timestamp(r.crs_arr_ts))] = np.asarray(r.grid)


def precompute(queries) -> None:
    """Predict quantile grids for many (origin, carrier, crs_arr_ts) in one TabPFN call."""
    _load_cache()
    todo = list(dict.fromkeys(k for k in (_key(*q) for q in queries) if k not in _mem))
    if not todo:
        return
    enc, m = _regressor()
    X = np.vstack([enc.numeric(features_from_query(o, c, t, _distance(o, c))) for o, c, t in todo])
    q = m.predict(X, output_type="quantiles", quantiles=[float(x) for x in GRID])
    grids = np.maximum.accumulate(np.stack(q, axis=1), axis=1)
    _mem.update(zip(todo, grids))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"origin": [k[0] for k in _mem], "carrier": [k[1] for k in _mem],
                  "crs_arr_ts": [k[2] for k in _mem],
                  "grid": [list(map(float, g)) for g in _mem.values()]}).to_parquet(CACHE)


def flight_arrival_samples(origin: str, carrier: str, crs_arr_ts: datetime,
                           n: int = 2000, seed: int = SEED) -> np.ndarray:
    """n sampled ArrDelay minutes, conditional on the flight not being cancelled."""
    k = _key(origin, carrier, crs_arr_ts)
    _load_cache()
    if k not in _mem:
        precompute([k])
    u = np.random.default_rng(seed).uniform(GRID[0], GRID[-1], size=n)
    return np.interp(u, GRID, _mem[k])


@functools.lru_cache(maxsize=1)
def _cancel_model() -> CancelModel:
    return CancelModel(load_months(TRAIN_MONTHS))


def p_cancel(origin: str, carrier: str, crs_arr_ts: datetime) -> float:
    """Shrunken empirical cancellation rate for the origin x carrier cell (see cancel.py).
    crs_arr_ts is accepted for the interface but not used."""
    return _cancel_model()(origin, carrier)


if __name__ == "__main__":  # smoke test: flock /tmp/snooze-oracle-gpu.lock python -m src.flight.model
    import time

    t0 = time.time()
    qs = [("BQN", "B6", "2026-10-26 05:10"), ("ATL", "DL", "2026-10-26 07:00"),
          ("XXX", "ZZ", "2026-10-26 06:00")]
    precompute(qs)
    print(f"precompute {len(qs)} queries: {time.time() - t0:.1f}s", flush=True)
    for q in qs:
        s = flight_arrival_samples(*q)
        print(q, "n", len(s), "P10/P50/P90", np.percentile(s, [10, 50, 90]).round(1),
              "p_cancel", round(p_cancel(*q), 4))
