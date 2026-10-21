"""Round 2 scoring by route density, with day-level bootstrap intervals.

Usage (from the repo root, after r2_data, r2_baselines and r2_tabpfn):
    python -m src.ride.r2_evaluate

Writes data/ride2/results_table.md / .csv (model x density bin) and
data/ride2/paired_diffs.md / .csv (TabPFN minus each baseline, same test rows).

Bootstrap: N_BOOT replicates; each draws the 31 August days with replacement and keeps
every test row of a drawn day (as many times as the day is drawn). A metric is
sum(weight * value) / sum(weight) over the drawn rows. The same day draws are used for
every model and bin. Intervals are 2.5/97.5 percentiles.
Bins use weight 1 per row (within a bin, dense bins were sampled at the same rate every
day). The "all (weighted)" row uses weight = pool rows / sampled rows of the row's
(bin, day) stratum, so it estimates the full August test pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import QUANTILES, TARGET, qcol
from .evaluate import pinball
from .r2_data import DENSITY_BINS
from .retrieval import OUT2, load_train

MODELS = ["fixed_buffer", "lookup", "lgbm_quantile", "tabpfn"]
BASELINES = ["fixed_buffer", "lookup", "lgbm_quantile"]
N_BOOT = 2000
SEED = 0


def per_row(y: np.ndarray, p: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "p90_coverage": (y <= p["q90"].to_numpy()).astype(float),
        "pinball_q90": pinball(y, p["q90"].to_numpy(), 0.9),
        "pinball_mean": np.mean([pinball(y, p[qcol(q)].to_numpy(), q) for q in QUANTILES],
                                axis=0),
        "mean_q90": p["q90"].to_numpy(float),
    }


class DayBoot:
    def __init__(self, day: np.ndarray, n_boot: int, seed: int):
        self.days = np.unique(day)
        self.day_idx = np.searchsorted(self.days, day)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.days), (n_boot, len(self.days)))
        # counts[b, d] = how many times day d is drawn in replicate b
        self.counts = np.zeros((n_boot, len(self.days)))
        np.add.at(self.counts, (np.arange(n_boot)[:, None], draws), 1)

    def __call__(self, x: np.ndarray, w: np.ndarray, mask: np.ndarray):
        nd = len(self.days)
        S = np.bincount(self.day_idx[mask], weights=(w * x)[mask], minlength=nd)
        N = np.bincount(self.day_idx[mask], weights=w[mask], minlength=nd)
        est = S.sum() / N.sum()
        boot = (self.counts @ S) / (self.counts @ N)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return est, lo, hi


def train_rows_summary(model: str, rows: pd.DataFrame, n_train: int) -> str:
    if model == "fixed_buffer":
        return "0"
    if model == "lgbm_quantile":
        return f"{n_train:,} (all)"
    col = "lookup_cell_rows" if model == "lookup" else "n_ctx"
    v = rows[col].to_numpy()
    return f"median {np.median(v):,.0f} [{v.min():,}-{v.max():,}]"


def main() -> None:
    test = pd.read_parquet(OUT2 / "test.parquet")
    n_train = len(load_train())
    y = test[TARGET].to_numpy()
    tab_stats = pd.read_json(OUT2 / "tabpfn_stats.jsonl", lines=True).drop_duplicates(
        "ctx_key", keep="last")
    # Lines written before the CPU fallback existed have no device field: all GPU.
    tab_stats["device"] = tab_stats.get("device", pd.Series(index=tab_stats.index,
                                                             dtype=object)).fillna("cuda")
    test = test.merge(tab_stats[["ctx_key", "n_ctx", "n_match", "capped", "device"]],
                      on="ctx_key",
                      how="left", validate="many_to_one")
    lk = pd.read_parquet(OUT2 / "preds_lookup.parquet")
    test["lookup_cell_rows"] = lk["lookup_cell_rows"].to_numpy()
    test["lookup_level"] = lk["lookup_level"].to_numpy()

    vals = {}
    for m in MODELS:
        p = pd.read_parquet(OUT2 / f"preds_{m}.parquet")
        assert (p["test_id"].to_numpy() == test["test_id"].to_numpy()).all(), m
        vals[m] = per_row(y, p)

    boot = DayBoot(test["day"].to_numpy(), N_BOOT, SEED)
    ones = np.ones(len(test))
    splits = [(b, (test["density"] == b).to_numpy(), ones) for b in DENSITY_BINS]
    splits.append(("all (weighted to pool)", np.ones(len(test), bool),
                   test["weight"].to_numpy()))

    def fmt(e, lo, hi, d):
        return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"

    rows, raw = [], []
    for split, mask, w in splits:
        for m in MODELS:
            r = {"density": split, "model": m, "n_test": int(mask.sum()),
                 "train_rows": train_rows_summary(m, test[mask], n_train)}
            for metric, d in [("p90_coverage", 3), ("pinball_q90", 2), ("pinball_mean", 2),
                              ("mean_q90", 1)]:
                e, lo, hi = boot(vals[m][metric], w, mask)
                r[metric] = fmt(e, lo, hi, d)
                raw.append({"density": split, "model": m, "metric": metric,
                            "est": e, "lo": lo, "hi": hi})
            rows.append(r)
    res = pd.DataFrame(rows)

    drows = []
    for split, mask, w in splits:
        for b in BASELINES:
            r = {"density": split, "comparison": f"tabpfn - {b}", "n_test": int(mask.sum())}
            for metric, d in [("p90_coverage", 3), ("pinball_q90", 2), ("pinball_mean", 2)]:
                e, lo, hi = boot(vals["tabpfn"][metric] - vals[b][metric], w, mask)
                r[metric] = fmt(e, lo, hi, d)
                raw.append({"density": split, "model": f"tabpfn-{b}", "metric": metric,
                            "est": e, "lo": lo, "hi": hi})
            drows.append(r)
    diffs = pd.DataFrame(drows)

    # TabPFN context sizes per bin (per test row and per context).
    ctx_rows = []
    for split, mask, _ in splits:
        t = test[mask]
        ctx_rows.append({
            "density": split, "n_test": len(t), "contexts": t["ctx_key"].nunique(),
            "ctx_rows_p10/p50/p90 (per sampled test row)":
                "/".join(f"{v:,.0f}" for v in np.percentile(t["n_ctx"], [10, 50, 90])),
            "widened_to_borough": f"{(t['ctx_level'] == 'borough').mean():.1%}",
            "hit_10k_cap": f"{t['capped'].mean():.1%}",
            "tabpfn_on_cpu": f"{(t['device'] == 'cpu').mean():.1%}",
            "lookup_full_cell": f"{(t['lookup_level'] == 0).mean():.1%}",
        })
    ctx = pd.DataFrame(ctx_rows)

    pd.DataFrame(raw).to_csv(OUT2 / "results_raw.csv", index=False)
    for name, df in [("results_table", res), ("paired_diffs", diffs), ("context_sizes", ctx)]:
        df.to_csv(OUT2 / f"{name}.csv", index=False)
        md = to_md(df)
        (OUT2 / f"{name}.md").write_text(md + "\n")
        print(md + "\n")


def to_md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
