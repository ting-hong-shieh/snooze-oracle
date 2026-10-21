"""`ride_total_samples`: draws from TabPFN-3.5's predictive distribution of total_min.

The model is the same one evaluated in RESULTS.md: TabPFNRegressor with library defaults,
fit on data/ride/context.parquet (10,000 July 2026 trips). It is fit lazily on first use
and kept in memory. Run anything that calls this on the GPU under the shared lock:
    flock /tmp/snooze-oracle-gpu.lock python ...
Set SNOOZE_RIDE_DEVICE=cpu to run on the CPU instead (slow; only for small tests).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from .common import FEATURES, load_context

_MODEL = None
MIN_TOTAL_MIN = 1.0  # processed table keeps ride_min >= 1, so total_min >= 1


def _model():
    global _MODEL
    if _MODEL is None:
        from .tabpfn_model import fit_model

        _MODEL = fit_model(load_context(), device=os.environ.get("SNOOZE_RIDE_DEVICE", "cuda"))
    return _MODEL


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
    levels = ((np.arange(n) + 0.5) / n).tolist()
    qs = _model().predict(features(pu_zone, do_zone, request_ts),
                          output_type="quantiles", quantiles=levels)
    samples = np.maximum(np.array([v[0] for v in qs], dtype=np.float64), MIN_TOTAL_MIN)
    (rng or np.random.default_rng()).shuffle(samples)
    return samples
