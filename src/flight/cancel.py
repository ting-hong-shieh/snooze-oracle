"""p_cancel: shrunken empirical cancellation rate, and its calibration on 2026-08.

Model (hierarchical shrinkage of counts, no fitting beyond counting):
    p_global  = cancelled / scheduled over all training rows (03:00-08:59 arrivals)
    p_carrier = (c_carrier + K * p_global)  / (n_carrier + K)
    p_oc      = (c_oc      + K * p_carrier) / (n_oc      + K)        # origin x carrier
with K = 100 pseudo-flights. Unknown carriers fall back to p_global, unknown cells to
p_carrier. Diverted flights count as not cancelled.

Round 2: when `dest` is given, one more level, shrunk to the origin x carrier rate:
    p_ocd     = (c_ocd     + K * p_oc)      / (n_ocd     + K)        # origin x carrier x dest
With dest=None the round-1 value is returned unchanged.

    python -m src.flight.cancel        # writes data/flight/cancel_calibration.json (round 1)
    python -m src.flight.cancel 2      # writes data/flight2/cancel_calibration2.json (round 2)
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

        g = train_all.assign(c=c).groupby(["origin", "carrier", "dest"])["c"].agg(["sum", "size"])
        self.cells_d = {key: (s, n) for key, s, n in zip(g.index, g["sum"], g["size"])}

    def __call__(self, origin: str, carrier: str, dest: str | None = None) -> float:
        pc = self.p_carrier.get(carrier, self.p_global)
        s, n = self.cells.get((origin, carrier), (0.0, 0))
        p = (s + self.k * pc) / (n + self.k)
        if dest is not None:
            s, n = self.cells_d.get((origin, carrier, dest), (0.0, 0))
            p = (s + self.k * p) / (n + self.k)
        return float(p)


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


def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    eps = 1e-6
    return {"mean_pred": round(float(p.mean()), 4), "brier": round(float(np.mean((p - y) ** 2)), 5),
            "logloss": round(float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))), 4)}


def calibration2() -> dict:
    """Round 2 (report only; p_cancel's design is unchanged apart from the dest level):
    (a) does the dest level help, (b) do the previous-day regime features carry signal
    about cancellations? (b) fits a logistic regression on the training rows with
    logit(p_cancel) as a feature plus the two regime features (missing -> training median),
    and scores it on 2026-08 next to p_cancel."""
    from sklearn.linear_model import LogisticRegression

    from .regime import REGIME_FEATURES, add_regime

    train_all = add_regime(load_months(TRAIN_MONTHS))
    test_all = add_regime(load_months(TEST_MONTHS))
    m = CancelModel(train_all)
    y = test_all["cancelled"].astype(float).to_numpy()
    p_oc = np.array([m(o, c) for o, c in zip(test_all["origin"], test_all["carrier"])])
    p_ocd = np.array([m(o, c, d) for o, c, d in
                      zip(test_all["origin"], test_all["carrier"], test_all["dest"])])
    out = {"test_rows": int(len(y)), "observed_rate": round(float(y.mean()), 4),
           "constant_train_rate": _scores(y, np.full_like(y, m.p_global)),
           "p_cancel_round1 (origin x carrier)": _scores(y, p_oc),
           "p_cancel_with_dest (origin x carrier x dest)": _scores(y, p_ocd)}

    def design(df, pp):
        X = df[REGIME_FEATURES].fillna(med).to_numpy()
        lp = np.log(np.clip(pp, 1e-4, 1 - 1e-4) / (1 - np.clip(pp, 1e-4, 1 - 1e-4)))
        return np.column_stack([lp, X])

    med = train_all[REGIME_FEATURES].median()
    ytr = train_all["cancelled"].astype(float).to_numpy()
    p_tr = np.array([m(o, c, d) for o, c, d in
                     zip(train_all["origin"], train_all["carrier"], train_all["dest"])])
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(design(train_all, p_tr), ytr)
    out["logistic_with_regime (report only)"] = _scores(y, lr.predict_proba(design(test_all, p_ocd))[:, 1])
    out["logistic_coefs [logit p, origin_prev, dest_prev]"] = [round(float(v), 4) for v in lr.coef_[0]]
    # Cancellation rate by regime of day A-2 at the destination (training-day quartiles).
    from .regime import _table
    t = _table()
    t = t[(t.kind == "dest_arr") & (t.n_flights > 0)]
    td = pd.to_datetime(t.day)
    tr = t[(td >= "2025-08-01") & (td < "2026-08-01")]["mean_delay"]
    q25, q75 = float(tr.quantile(0.25)), float(tr.quantile(0.75))
    rows = {}
    for name, df in [("train", train_all), ("test", test_all)]:
        x = df["dest_prev_arr_delay"]
        g = np.where(x >= q75, "bad", np.where(x < q25, "calm", np.where(x.isna(), "missing", "middle")))
        r = df.assign(g=g, c=df["cancelled"].astype(float)).groupby("g")["c"].agg(["mean", "size"])
        rows[name] = {k: {"cancel_rate": round(float(v["mean"]), 4), "n": int(v["size"])}
                      for k, v in r.iterrows()}
    out["cancel_rate_by_A-2_dest_regime"] = {"thresholds_min": [round(q25, 1), round(q75, 1)], **rows}
    return out


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["2"]:
        from .regime import OUT2
        res = calibration2()
        OUT2.mkdir(parents=True, exist_ok=True)
        (OUT2 / "cancel_calibration2.json").write_text(json.dumps(res, indent=2))
    else:
        res = calibration()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "cancel_calibration.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
