"""Round 3 retrieval context (SPEC "Ride TabPFN context"), one per (route, request hour).

For a query on route pu_zone -> do_zone requested in hour h:
  1. every July row of that route with req_hour in [h-2, h+2] (truncated to 03-08);
  2. if fewer than MAX_CONTEXT_ROWS, the remaining slots are filled with rows of the same
     pu_borough -> do_zone in the same hours (own-route rows excluded), sampled uniformly
     without replacement, seeded by the key; total capped at MAX_CONTEXT_ROWS.
The same context is used by TabPFN and, for the retrieval control, by the lookup table and
LightGBM.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .retrieval import MAX_CONTEXT_ROWS, OUT2, hour_band, load_train, zone_borough

OUT3 = OUT2.parent / "ride3"


def key_name(pu_zone: int, do_zone: int, h: int) -> str:
    return f"r3_{int(pu_zone)}_{int(do_zone)}_{int(h)}"


def context_rows(pu_zone: int, do_zone: int, h: int, train: pd.DataFrame | None = None,
                 hours: range | None = None) -> tuple[pd.DataFrame, int, int]:
    """(context, own-route rows, borough fill rows). `train` and `hours` (the request-hour
    window the band is truncated to) default to the airport setting."""
    tr = load_train() if train is None else train
    lo, hi = hour_band(h) if hours is None else (max(h - 2, hours.start),
                                                 min(h + 2, hours.stop - 1))
    band = tr["req_hour"].between(lo, hi) & (tr["do_zone"] == do_zone)
    own_m = band & (tr["pu_zone"] == pu_zone)
    own = tr[own_m]
    rng = np.random.default_rng(zlib.crc32(key_name(pu_zone, do_zone, h).encode()))
    if len(own) >= MAX_CONTEXT_ROWS:  # does not happen in July 2026 (max ~3.6k)
        own = own.iloc[np.sort(rng.choice(len(own), MAX_CONTEXT_ROWS, replace=False))]
        return own, len(own), 0
    pool = tr[band & ~own_m & (tr["pu_borough"] == zone_borough()[int(pu_zone)])]
    k = min(MAX_CONTEXT_ROWS - len(own), len(pool))
    fill = pool.iloc[np.sort(rng.choice(len(pool), k, replace=False))]
    return pd.concat([own, fill]), len(own), len(fill)
