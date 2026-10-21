"""Score every model's quantile predictions on the 2026-08 test sample.

Usage (from the repo root):
    python -m src.ride.evaluate

Reads data/ride/test.parquet and data/ride/preds/<model>.parquet; prints a Markdown table
and writes it to data/ride/results_table.md (and the numbers to results_table.csv).

Metrics per (model, split):
    p90_coverage   share of test rows with total_min <= predicted q90 (target 0.90)
    pinball_mean   pinball loss averaged over QUANTILES (0.1, 0.25, 0.5, 0.75, 0.9), minutes
    pinball_q90    pinball loss at 0.9 only, minutes
    mean_q90       average predicted q90 (how many minutes the model tells you to allow)
    ci95           95% bootstrap interval (1,000 row resamples) of pinball_mean
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT, PRED, QUANTILES, TARGET, load_test, qcol

MODELS = ["fixed_buffer", "lookup", "lgbm_quantile", "tabpfn"]
REFERENCE = ["lookup_full_july"]
N_BOOT = 1000


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> np.ndarray:
    d = y - pred
    return np.maximum(q * d, (q - 1) * d)


def score(y: np.ndarray, p: pd.DataFrame, rng: np.random.Generator) -> dict:
    per_row = np.mean([pinball(y, p[qcol(q)].to_numpy(), q) for q in QUANTILES], axis=0)
    boot = [per_row[rng.integers(0, len(y), len(y))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(y),
        "p90_coverage": float(np.mean(y <= p["q90"].to_numpy())),
        "pinball_mean": float(per_row.mean()),
        "ci95": f"{lo:.2f}-{hi:.2f}",
        "pinball_q90": float(pinball(y, p["q90"].to_numpy(), 0.9).mean()),
        "mean_q90": float(p["q90"].mean()),
    }


def to_markdown(df: pd.DataFrame) -> str:
    fmt = {"n": "{:,}".format, "p90_coverage": "{:.3f}".format,
           "pinball_mean": "{:.2f}".format, "pinball_q90": "{:.2f}".format,
           "mean_q90": "{:.1f}".format}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, str)(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    test = load_test()
    splits = {
        "dense": test["density"] == "dense",
        "sparse": test["density"] == "sparse",
        "all (50/50 strata)": np.ones(len(test), bool),
        "early 03:00-06:59, dense": (test["density"] == "dense") & test["req_hour"].between(3, 6),
        "early 03:00-06:59, sparse": (test["density"] == "sparse") & test["req_hour"].between(3, 6),
    }
    rows = []
    for model in MODELS + REFERENCE:
        p = pd.read_parquet(PRED / f"{model}.parquet")
        assert (p["test_id"].to_numpy() == test["test_id"].to_numpy()).all(), model
        label = model + (" (reference, all July rows)" if model in REFERENCE else "")
        for split, mask in splits.items():
            mask = np.asarray(mask)
            rng = np.random.default_rng(0)
            rows.append({"split": split, "model": label,
                         **score(test.loc[mask, TARGET].to_numpy(), p[mask], rng)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results_table.csv", index=False)
    md = to_markdown(res)
    (OUT / "results_table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
