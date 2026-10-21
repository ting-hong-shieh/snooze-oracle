"""Shared constants and helpers for the ride workstream."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "ride"
PRED = OUT / "preds"

FEATURES = ["pu_zone", "do_zone", "req_min_of_day", "dow", "is_weekend"]
CATEGORICAL = ["pu_zone", "do_zone"]  # zone ids are labels, not magnitudes
TARGET = "total_min"

# Quantile levels every model predicts; pinball loss is averaged over these.
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def qcol(q: float) -> str:
    return f"q{round(q * 100):02d}"


def load_context() -> pd.DataFrame:
    return pd.read_parquet(OUT / "context.parquet")


def load_test() -> pd.DataFrame:
    return pd.read_parquet(OUT / "test.parquet")


def save_pred(name: str, test_id: np.ndarray, qpred: dict[float, np.ndarray]) -> Path:
    PRED.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"test_id": test_id, **{qcol(q): qpred[q] for q in QUANTILES}})
    path = PRED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path
