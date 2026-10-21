"""Round 3 scoring: retrieval control (step 1) and the GPU-only TabPFN rerun (step 2).

Usage (from the repo root, after r3_control and r3_tabpfn run + merge):
    python -m src.ride.r3_evaluate

Same test rows, metrics and day-level bootstrap as round 2 (r2_evaluate). Writes
data/ride3/control.md and data/ride3/rerun.md.

Step 1, bins <30 and 30-199: TabPFN (r3 context) vs the lookup and LightGBM fit on the same
r3 context, with the round-2 full-data lookup and LightGBM as references.
Step 2, every bin: TabPFN r3 (GPU only, own route first) vs TabPFN round 2 (mixed devices,
round-2 context rule), plus the round-2 full-data baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import TARGET
from .r2_data import DENSITY_BINS
from .r2_evaluate import N_BOOT, SEED, DayBoot, per_row, to_md
from .r3_context import OUT3
from .retrieval import OUT2

METRICS = [("p90_coverage", 3), ("pinball_q90", 2), ("pinball_mean", 2)]


def load(name: str, test: pd.DataFrame) -> dict:
    path = {"tabpfn_r3": OUT3 / "preds_tabpfn.parquet",
            "lookup_ctx": OUT3 / "preds_lookup_ctx.parquet",
            "lgbm_ctx": OUT3 / "preds_lgbm_ctx.parquet",
            "tabpfn_r2": OUT2 / "preds_tabpfn.parquet",
            "lookup_full": OUT2 / "preds_lookup.parquet",
            "lgbm_full": OUT2 / "preds_lgbm_quantile.parquet"}[name]
    p = pd.read_parquet(path).set_index("test_id").loc[test["test_id"]].reset_index()
    return per_row(test[TARGET].to_numpy(), p)


def fmt(e, lo, hi, d):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def tables(test, vals, models, pairs, splits, boot):
    rows, drows = [], []
    for split, mask, w in splits:
        for m in models:
            r = {"density": split, "model": m, "n_test": int(mask.sum())}
            for metric, d in METRICS:
                r[metric] = fmt(*boot(vals[m][metric], w, mask), d)
            rows.append(r)
        for a, b in pairs:
            r = {"density": split, "comparison": f"{a} - {b}", "n_test": int(mask.sum())}
            for metric, d in METRICS:
                r[metric] = fmt(*boot(vals[a][metric] - vals[b][metric], w, mask), d)
            drows.append(r)
    return to_md(pd.DataFrame(rows)), to_md(pd.DataFrame(drows))


def main() -> None:
    test = pd.read_parquet(OUT2 / "test.parquet")
    boot = DayBoot(test["day"].to_numpy(), N_BOOT, SEED)
    ones = np.ones(len(test))

    # step 1
    sparse = test["density"].isin(["<30", "30-199"]).to_numpy()
    ts = test[sparse].reset_index(drop=True)
    bs = DayBoot(ts["day"].to_numpy(), N_BOOT, SEED)
    models = ["tabpfn_r3", "lookup_ctx", "lgbm_ctx", "lookup_full", "lgbm_full"]
    vals = {m: load(m, ts) for m in models}
    o = np.ones(len(ts))
    splits = [(b, (ts["density"] == b).to_numpy(), o) for b in ("<30", "30-199")]
    pairs = [("tabpfn_r3", "lookup_ctx"), ("tabpfn_r3", "lgbm_ctx"),
             ("lookup_ctx", "lookup_full"), ("lgbm_ctx", "lgbm_full"),
             ("tabpfn_r3", "lookup_full"), ("tabpfn_r3", "lgbm_full")]
    t1, d1 = tables(ts, vals, models, pairs, splits, bs)
    lk = pd.read_parquet(OUT3 / "preds_lookup_ctx.parquet")
    lv = np.bincount(lk["lookup_level"], minlength=4)
    note = (f"Lookup on the r3 context, back-off level used per test row "
            f"(cell / route / pickup zone / whole context): {lv.tolist()}")
    (OUT3 / "control.md").write_text("\n\n".join([t1, d1, note]) + "\n")
    print(t1, "\n\n", d1, "\n\n", note)

    # step 2
    models = ["tabpfn_r3", "tabpfn_r2", "lookup_full", "lgbm_full"]
    vals = {m: load(m, test) for m in models}
    splits = [(b, (test["density"] == b).to_numpy(), ones) for b in DENSITY_BINS]
    splits.append(("all (weighted to pool)", np.ones(len(test), bool), test["weight"].to_numpy()))
    pairs = [("tabpfn_r3", "tabpfn_r2"), ("tabpfn_r3", "lookup_full"), ("tabpfn_r3", "lgbm_full")]
    t2, d2 = tables(test, vals, models, pairs, splits, boot)
    st = pd.read_json(OUT3 / "tabpfn_stats.jsonl", lines=True).drop_duplicates("key", keep="last")
    assert (st["device"] == "cuda").all()
    note = (f"TabPFN r3: {len(st)} contexts, all on cuda; context rows median "
            f"{st['n_ctx'].median():,.0f} (min {st['n_ctx'].min():,}), own-route rows median "
            f"{st['n_own'].median():,.0f}; {int((st['n_ctx'] < 10_000).sum())} contexts under "
            f"10,000 rows; GPU fit+predict {st['fit_s'].sum() + st['predict_s'].sum():,.0f} s, "
            f"peak alloc {st['peak_alloc_mib'].max():,} MiB, peak RSS {st['rss_mib'].max():,} MiB")
    (OUT3 / "rerun.md").write_text("\n\n".join([t2, d2, note]) + "\n")
    print("\n\n", t2, "\n\n", d2, "\n\n", note)


if __name__ == "__main__":
    main()
