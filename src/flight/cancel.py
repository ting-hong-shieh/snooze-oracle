"""p_cancel: shrunken empirical cancellation rate, and its calibration on 2026-08.

Model (hierarchical shrinkage of counts, no fitting beyond counting):
    p_global  = cancelled / scheduled over all training rows (03:00-08:59 arrivals)
    p_carrier = (c_carrier + K * p_global)  / (n_carrier + K)
    p_oc      = (c_oc      + K * p_carrier) / (n_oc      + K)        # origin x carrier
with K = 100 pseudo-flights. Unknown carriers fall back to p_global, unknown cells to
p_carrier. Diverted flights count as not cancelled.

    python -m src.flight.cancel        # writes data/flight/cancel_calibration.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import OUT, TEST_MONTHS, TRAIN_MONTHS, load_months

K = 100.0


class CancelModel:
    def __init__(self, train_all: pd.DataFrame, k: float = K):
        c = train_all["cancelled"].astype(float)
        self.k = k
        self.p_global = float(c.mean())
        g = train_all.assign(c=c).groupby("carrier")["c"].agg(["sum", "size"])
        self.p_carrier = ((g["sum"] + k * self.p_global) / (g["size"] + k)).to_dict()
        g = train_all.assign(c=c).groupby(["origin", "carrier"])["c"].agg(["sum", "size"])
        self.cells = {key: (s, n) for key, s, n in zip(g.index, g["sum"], g["size"])}

    def __call__(self, origin: str, carrier: str) -> float:
        pc = self.p_carrier.get(carrier, self.p_global)
        s, n = self.cells.get((origin, carrier), (0.0, 0))
        return float((s + self.k * pc) / (n + self.k))


def calibration() -> dict:
    train_all, test_all = load_months(TRAIN_MONTHS), load_months(TEST_MONTHS)
    m = CancelModel(train_all)
    y = test_all["cancelled"].astype(float).to_numpy()
    p = np.array([m(o, c) for o, c in zip(test_all["origin"], test_all["carrier"])])
    eps = 1e-6
    ll = lambda pp: float(-np.mean(y * np.log(pp + eps) + (1 - y) * np.log(1 - pp + eps)))
    base = np.full_like(p, m.p_global)
    edges = np.quantile(p, [0, 0.25, 0.5, 0.75, 1.0])
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 3)
    rel = [{"bin": i, "p_range": [round(float(p[b == i].min()), 4), round(float(p[b == i].max()), 4)],
            "n": int((b == i).sum()), "mean_pred": round(float(p[b == i].mean()), 4),
            "observed": round(float(y[b == i].mean()), 4)} for i in range(4)]
    by_day = test_all.assign(c=y).groupby(test_all["crs_arr_ts"].dt.date)["c"].agg(["sum", "size"])
    out = {
        "K": K, "test_rows": int(len(y)), "observed_rate": round(float(y.mean()), 4),
        "mean_pred": round(float(p.mean()), 4), "train_rate": round(m.p_global, 4),
        "brier": round(float(np.mean((p - y) ** 2)), 5),
        "brier_constant_train_rate": round(float(np.mean((base - y) ** 2)), 5),
        "logloss": round(ll(p), 4), "logloss_constant_train_rate": round(ll(base), 4),
        "reliability_quartiles": rel,
        "top_days": [{"date": str(d), "cancelled": int(r["sum"]), "scheduled": int(r["size"])}
                     for d, r in by_day.sort_values("sum", ascending=False).head(5).iterrows()],
        "monthly_train_rates": {mo: round(float(load_months([mo])["cancelled"].mean()), 4)
                                for mo in TRAIN_MONTHS},
    }
    return out


if __name__ == "__main__":
    res = calibration()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cancel_calibration.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
