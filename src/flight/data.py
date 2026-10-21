"""Shared data loading for the flight workstream (JFK/LGA arrivals scheduled 03:00-08:59).

Every model reads its rows from here so that all models see exactly the same training rows
and are scored on exactly the same test rows.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "flight"

TRAIN_MONTHS = [f"2025-{m:02d}" for m in range(8, 13)] + [f"2026-{m:02d}" for m in range(1, 8)]
TEST_MONTHS = ["2026-08"]
WINDOW = (300, 859)  # crs_arr_hhmm, inclusive

FEATURES = ["origin", "carrier", "crs_arr_min_of_day", "dow", "month", "distance"]
CAT_FEATURES = ["origin", "carrier"]

# Density bins: training flights (flown, 12 months, 03:00-08:59) per origin x carrier.
DENSITY_BINS = [(0, 0, "0 (unseen)"), (1, 49, "1-49"), (50, 199, "50-199"),
                (200, 799, "200-799"), (800, 10**9, "800+")]

SEED = 0


def load_months(months: list[str]) -> pd.DataFrame:
    files = [str(PROCESSED / f"flights_{m}.parquet") for m in months]
    lo, hi = WINDOW
    df = duckdb.connect().execute(
        f"SELECT * FROM read_parquet({files!r}) WHERE crs_arr_hhmm BETWEEN {lo} AND {hi}"
    ).df()
    df = df.rename(columns={"airline": "carrier"})
    ts = pd.to_datetime(df["crs_arr_ts"])
    df["dow"] = ts.dt.dayofweek.astype(int)      # from scheduled arrival (NYC local)
    df["month"] = ts.dt.month.astype(int)
    df["flown"] = ~df["cancelled"] & ~df["diverted"] & df["arr_delay"].notna()
    return df.sort_values(["crs_arr_ts", "carrier", "flight_num", "origin"]).reset_index(drop=True)


def features_from_query(origin: str, carrier: str, crs_arr_ts, distance: float) -> pd.DataFrame:
    ts = pd.Timestamp(crs_arr_ts)
    return pd.DataFrame({
        "origin": [origin], "carrier": [carrier],
        "crs_arr_min_of_day": [ts.hour * 60 + ts.minute],
        "dow": [ts.dayofweek], "month": [ts.month], "distance": [float(distance)],
    })


# Round 2: dest (JFK/LGA, categorical) and the two previous-day regime features (regime.py).
FEATURES_REGIME = FEATURES + ["dest", "origin_prev_dep_delay", "dest_prev_arr_delay"]
CAT_FEATURES_REGIME = CAT_FEATURES + ["dest"]


class Encoder:
    """Integer codes for the categorical columns learned on the training rows; unseen -> NaN.
    Defaults are the round-1 feature set."""

    def __init__(self, train: pd.DataFrame, features: list[str] = FEATURES,
                 cats: list[str] = CAT_FEATURES):
        self.features, self.cats = list(features), list(cats)
        self.maps = {c: {v: i for i, v in enumerate(sorted(train[c].unique()))} for c in self.cats}

    @property
    def cat_indices(self) -> list[int]:
        return [self.features.index(c) for c in self.cats]

    def numeric(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].copy()
        for c in self.cats:
            X[c] = X[c].map(self.maps[c]).astype(float)
        return X.astype(float).to_numpy()

    def pandas_cat(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features].copy()
        for c in self.cats:
            X[c] = pd.Categorical(X[c], categories=sorted(self.maps[c]))
        return X


def density_label(n: int) -> str:
    for lo, hi, lab in DENSITY_BINS:
        if lo <= n <= hi:
            return lab
    raise ValueError(n)


def split(context_size: int | None = None):
    """Return (train_flown, test_flown, train_all, test_all).

    train_flown is the delay-model training set shared by every model: all flown training
    rows, or a seeded uniform subsample of `context_size` rows if given.
    """
    train_all = load_months(TRAIN_MONTHS)
    test_all = load_months(TEST_MONTHS)
    train = train_all[train_all["flown"]].reset_index(drop=True)
    if context_size is not None and context_size < len(train):
        train = train.sample(n=context_size, random_state=SEED).sort_index().reset_index(drop=True)
    test = test_all[test_all["flown"]].reset_index(drop=True)
    counts = train.groupby(["origin", "carrier"]).size()
    key = list(zip(test["origin"], test["carrier"]))
    test["n_train_oc"] = [int(counts.get(k, 0)) for k in key]
    test["density"] = test["n_train_oc"].map(density_label)
    return train, test, train_all, test_all
