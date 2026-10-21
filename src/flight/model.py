"""Runtime interface (see SPEC.md "Interfaces"):

    flight_arrival_samples(origin, carrier, crs_arr_ts, n=2000, dest=None) -> np.ndarray
    p_cancel(origin, carrier, crs_arr_ts, dest=None) -> float

Round 2. flight_arrival_samples uses TabPFN-3.5 with the round-2 "dest only" feature set
(round-1 features + dest) by default, fitted on the same context rows as the 2026-08
evaluation (all flown 2025-08 ... 2026-07 arrivals scheduled 03:00-08:59; nothing is pickled,
so no weights are written to disk). SNOOZE_FLIGHT_VARIANT=with switches to the variant that
also uses the two previous-day regime features (see regime.py and RESULTS_round2.md for why
it is not the default). It takes TabPFN's predictive distribution on a fine quantile grid
and draws n samples by inverse-CDF interpolation on that grid.

dest known: exact route distance (most common training distance for origin -> dest).
dest None: the flight is predicted for both JFK and LGA and the samples are a mixture, each
sample's airport drawn with the training share of the (origin, carrier) cell's flights to
JFK vs LGA (else the origin's share, else the overall share).

With SNOOZE_FLIGHT_VARIANT=with, regime features are read from
data/flight2/airport_day.parquet for the date two days before the scheduled arrival date.
That table is built from monthly BTS files, which are published weeks later, so for a real
upcoming morning they are missing (NaN) unless a live source fills the table; the model then
predicts with NaN, which it rarely saw in training.

One TabPFN predict call costs minutes on the shared GPU whether it is for 1 row or 500, so
call `precompute([...])` for all flights of interest in one batch the night before; grids
are stored in data/flight2/grid_cache_<variant>.parquet and flight_arrival_samples reads them from
there. A flight that is not in the cache is predicted on the spot.

precompute takes /tmp/snooze-oracle-gpu.lock itself, only around fit + predict, and frees
GPU memory before releasing it. Do NOT wrap the process in `flock` on the same file (it
would wait on itself). Set SNOOZE_FLIGHT_DEVICE=cpu to run on CPU instead (no lock).
"""

from __future__ import annotations

import functools
import os
from datetime import datetime

import numpy as np
import pandas as pd

from .cancel import CancelModel
from .data import SEED, TRAIN_MONTHS, Encoder, features_from_query, load_months, split
from .evaluate import GRID
from .evaluate2 import VARIANTS, gpu_lock
from .regime import OUT2, add_regime

VARIANT = os.environ.get("SNOOZE_FLIGHT_VARIANT", "dest only")
_REGIME = "origin_prev_dep_delay" in VARIANTS[VARIANT][0]
CACHE = OUT2 / f"grid_cache_{VARIANT.replace(' ', '_')}.parquet"
DESTS = ("JFK", "LGA")
_mem: dict[tuple, np.ndarray] = {}


@functools.lru_cache(maxsize=1)
def _train():
    train, _, _, _ = split(None)
    return add_regime(train) if _REGIME else train


def _distance(origin: str, carrier: str, dest: str) -> float:
    """Route distance from the training rows (origin -> dest), else round-1's guess."""
    t = _train()
    for d in (t[(t.origin == origin) & (t.dest == dest)],
              t[(t.origin == origin) & (t.carrier == carrier)], t[t.origin == origin]):
        if len(d):
            return float(d["distance"].mode().iloc[0])
    return float(t["distance"].median())


def _dest_share(origin: str, carrier: str) -> dict[str, float]:
    t = _train()
    for d in (t[(t.origin == origin) & (t.carrier == carrier)], t[t.origin == origin], t):
        if len(d):
            s = d["dest"].value_counts(normalize=True)
            return {k: float(s.get(k, 0.0)) for k in DESTS}
    raise AssertionError


def _key(origin: str, carrier: str, crs_arr_ts, dest: str) -> tuple:
    return (origin, carrier, pd.Timestamp(crs_arr_ts).floor("min"), dest)


def _load_cache() -> None:
    if not _mem and CACHE.exists():
        for r in pd.read_parquet(CACHE).itertuples(index=False):
            _mem[(r.origin, r.carrier, pd.Timestamp(r.crs_arr_ts), r.dest)] = np.asarray(r.grid)


def _expand(q) -> list[tuple]:
    origin, carrier, ts = q[:3]
    dest = q[3] if len(q) > 3 else None
    return [_key(origin, carrier, ts, d) for d in ((dest,) if dest else DESTS)]


def precompute(queries) -> None:
    """Predict quantile grids for many (origin, carrier, crs_arr_ts[, dest]) in one TabPFN
    call. A query without dest is predicted for both JFK and LGA."""
    _load_cache()
    todo = list(dict.fromkeys(k for q in queries for k in _expand(q) if k not in _mem))
    if not todo:
        return
    from tabpfn import TabPFNRegressor

    train = _train()
    enc = Encoder(train, *VARIANTS[VARIANT])
    rows = []
    for o, c, t, d in todo:
        f = features_from_query(o, c, t, _distance(o, c, d))
        f["dest"], f["crs_arr_ts"] = d, t
        rows.append(f)
    q_df = pd.concat(rows, ignore_index=True)
    X = enc.numeric(add_regime(q_df) if _REGIME else q_df)
    device = os.environ.get("SNOOZE_FLIGHT_DEVICE", "cuda")

    def fit_predict():
        m = TabPFNRegressor(device=device, categorical_features_indices=enc.cat_indices,
                            random_state=SEED)
        m.fit(enc.numeric(train), train["arr_delay"].to_numpy())
        q = m.predict(X, output_type="quantiles", quantiles=[float(x) for x in GRID])
        del m
        return q

    if device == "cpu":
        q = fit_predict()
    else:
        import torch
        with gpu_lock():
            q = fit_predict()
            torch.cuda.empty_cache()
    grids = np.maximum.accumulate(np.stack(q, axis=1), axis=1)
    _mem.update(zip(todo, grids))
    OUT2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"origin": [k[0] for k in _mem], "carrier": [k[1] for k in _mem],
                  "crs_arr_ts": [k[2] for k in _mem], "dest": [k[3] for k in _mem],
                  "grid": [list(map(float, g)) for g in _mem.values()]}).to_parquet(CACHE)


def flight_arrival_samples(origin: str, carrier: str, crs_arr_ts: datetime,
                           n: int = 2000, dest: str | None = None,
                           seed: int = SEED) -> np.ndarray:
    """n sampled ArrDelay minutes, conditional on the flight not being cancelled.
    `dest` is "JFK" or "LGA" when known; otherwise a JFK/LGA mixture (see module doc)."""
    q = (origin, carrier, crs_arr_ts, dest)
    _load_cache()
    keys = _expand(q)
    if any(k not in _mem for k in keys):
        precompute([q])
    rng = np.random.default_rng(seed)
    u = rng.uniform(GRID[0], GRID[-1], size=n)
    if dest:
        return np.interp(u, GRID, _mem[keys[0]])
    share = _dest_share(origin, carrier)
    pick = rng.choice(len(DESTS), size=n, p=[share[d] for d in DESTS])
    out = np.empty(n)
    for j, k in enumerate(keys):
        out[pick == j] = np.interp(u[pick == j], GRID, _mem[k])
    return out


@functools.lru_cache(maxsize=1)
def _cancel_model() -> CancelModel:
    return CancelModel(load_months(TRAIN_MONTHS))


def p_cancel(origin: str, carrier: str, crs_arr_ts: datetime, dest: str | None = None) -> float:
    """Shrunken empirical cancellation rate for the origin x carrier (x dest, when given)
    cell (see cancel.py). crs_arr_ts is accepted for the interface but not used."""
    return _cancel_model()(origin, carrier, dest)


if __name__ == "__main__":  # smoke test: python -m src.flight.model  (takes the GPU lock itself)
    import time

    from .evaluate2 import VARIANTS as _V

    # One realistic nightly batch: every flown test flight arriving 2026-08-26, plus a flight
    # without dest (JFK/LGA mixture) and an unknown origin/carrier. The batch is checked
    # against the evaluation's grid for the same rows (same variant, same context).
    _, test, _, _ = split(None)
    day = test[test["crs_arr_ts"].dt.date == pd.Timestamp("2026-08-26").date()]
    qs = [(r.origin, r.carrier, r.crs_arr_ts, r.dest) for r in day.itertuples()]
    extra = [("ATL", "DL", "2026-08-26 07:00", None), ("XXX", "ZZ", "2026-08-26 06:00", "LGA")]
    t0 = time.time()
    precompute(qs + extra)
    print(f"precompute {len(qs) + len(extra)} queries ({len(day)} test rows): "
          f"{time.time() - t0:.1f}s", flush=True)
    ev = np.load(OUT2 / f"tabpfn_{VARIANT.replace(' ', '_')}" / "grid.npy")[day.index]
    rt = np.stack([_mem[_key(*q)] for q in qs])
    print("max |runtime grid - evaluation grid| (min):", float(np.abs(rt - np.maximum.accumulate(ev, axis=1)).max()))
    for o, c, t, d in [qs[0], qs[-1]] + extra:
        s_ = flight_arrival_samples(o, c, t, dest=d)
        print((o, c, str(t), d), "n", len(s_), "P10/P50/P90", np.percentile(s_, [10, 50, 90]).round(1),
              "p_cancel", round(p_cancel(o, c, t, dest=d), 4))
