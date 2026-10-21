"""`ride_total_samples`: draws from TabPFN-3.5's predictive distribution of total_min.

Round 2 (see RESULTS_round2.md): the model is fit on the retrieval context of the query,
the same retrieval the round-2 evaluation scores (`retrieval.py`): July 2026 trips on the
same route within +-2 request hours, widened to the pickup borough when the route has
fewer than 500 such trips, capped at 10,000 rows. Only the product setting is supported:
do_zone 132 (JFK) or 138 (LaGuardia), request time 03:00-08:59.

The model for the most recent context is kept in memory, so a nightly batch should call
this grouped by route and hour. Run anything that calls this on the GPU under the lock:
    flock /tmp/snooze-oracle-gpu.lock python ...
Set SNOOZE_RIDE_DEVICE=cpu to run on the CPU instead (slow; only for small tests).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from .common import FEATURES
from .retrieval import DEST_ZONES, WINDOW_MIN, context_key, context_rows

_CACHE: dict = {}
MIN_TOTAL_MIN = 1.0  # processed table keeps ride_min >= 1, so total_min >= 1


def _model(key: tuple):
    if key not in _CACHE:
        from .r2_tabpfn import fit_context

        _CACHE.clear()  # one model at a time: GPU memory is shared
        ctx, _ = context_rows(key)
        _CACHE[key] = fit_context(ctx, device=os.environ.get("SNOOZE_RIDE_DEVICE", "cuda"))
    return _CACHE[key]


def features(pu_zone: int, do_zone: int, request_ts: datetime) -> np.ndarray:
    row = {
        "pu_zone": pu_zone,
        "do_zone": do_zone,
        "req_min_of_day": request_ts.hour * 60 + request_ts.minute,
        "dow": request_ts.weekday(),  # 0 = Monday, as in SPEC.md
        "is_weekend": int(request_ts.weekday() >= 5),
    }
    return np.array([[row[f] for f in FEATURES]], dtype=np.float64)


def ride_total_samples(pu_zone: int, do_zone: int, request_ts: datetime,
                       n: int = 2000, rng: np.random.Generator | None = None) -> np.ndarray:
    """n sampled total_min values (request -> drop-off, minutes) for an Uber requested
    at request_ts from pu_zone to do_zone.

    The values are TabPFN's predictive quantiles at levels (i + 0.5) / n, i = 0..n-1
    (inverse-CDF sampling on an even grid), clipped below at MIN_TOTAL_MIN, then put in a
    random order so they can be paired index-wise with other independent samples. Pass
    `rng` for a reproducible order; do not reuse one seeded order for two arrays you pair.
    """
    minute = request_ts.hour * 60 + request_ts.minute
    if do_zone not in DEST_ZONES or not WINDOW_MIN[0] <= minute <= WINDOW_MIN[1]:
        raise ValueError("ride_total_samples supports do_zone 132/138 and requests "
                         f"03:00-08:59 only, got do_zone={do_zone} at {request_ts:%H:%M}")
    key = context_key(int(pu_zone), int(do_zone), request_ts.hour)
    levels = ((np.arange(n) + 0.5) / n).tolist()
    qs = _model(key).predict(features(pu_zone, do_zone, request_ts),
                             output_type="quantiles", quantiles=levels)
    samples = np.maximum(np.array([v[0] for v in qs], dtype=np.float64), MIN_TOTAL_MIN)
    (rng or np.random.default_rng()).shuffle(samples)
    return samples
