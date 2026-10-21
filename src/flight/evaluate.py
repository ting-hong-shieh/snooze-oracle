"""Holdout comparison on 2026-08 (flights scheduled to arrive 03:00-08:59 at JFK/LGA).

Stages (run from the repo root; each writes to data/flight/):
    flock /tmp/snooze-oracle-gpu.lock python -m src.flight.evaluate tabpfn [--context N]
    python -m src.flight.evaluate baselines
    python -m src.flight.evaluate report

The tabpfn stage records the context rows it used (data/flight/context.json); the baselines
stage fits on exactly those rows.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

from .data import OUT, SEED, Encoder, split

QS = [0.1, 0.5, 0.9]
# Quantile grid kept from TabPFN's predictive distribution (used for sampling).
GRID = np.r_[0.001, 0.0025, np.round(np.arange(0.005, 0.9951, 0.005), 4), 0.9975, 0.999]
CHUNK = 500


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def run_tabpfn(context: int | None) -> None:
    import torch
    from tabpfn import TabPFNRegressor

    OUT.mkdir(parents=True, exist_ok=True)
    train, test, _, _ = split(context)
    enc = Encoder(train)
    Xtr, ytr = enc.numeric(train), train["arr_delay"].to_numpy()
    Xte = enc.numeric(test)
    print(f"context rows {len(train)}, test rows {len(test)}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = TabPFNRegressor(device="cuda", categorical_features_indices=[0, 1], random_state=SEED)
    model.fit(Xtr, ytr)
    t_fit = time.time() - t0
    print(f"fit {t_fit:.1f}s  peak alloc {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)
    parts = []
    for i in range(0, len(test), CHUNK):
        path = OUT / f"tabpfn_chunk_{i:05d}.npy"
        if path.exists():
            parts.append(np.load(path))
            continue
        q = model.predict(Xte[i:i + CHUNK], output_type="quantiles", quantiles=[float(x) for x in GRID])
        arr = np.stack(q, axis=1)  # rows x grid
        np.save(path, arr)
        parts.append(arr)
        print(f"  rows {i}-{i+len(arr)}  peak alloc {torch.cuda.max_memory_allocated()/2**20:.0f} MiB  "
              f"rss {_rss_gb():.2f} GB", flush=True)
    grid_q = np.concatenate(parts)
    np.save(OUT / "tabpfn_grid.npy", grid_q)
    meta = {
        "context_rows": len(train), "context_arg": context, "test_rows": len(test),
        "fit_s": round(t_fit, 1), "total_s": round(time.time() - t0, 1),
        "peak_gpu_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "peak_rss_gb": round(_rss_gb(), 2),
        "n_estimators": getattr(model, "n_estimators_", model.n_estimators),
        "tabpfn_model_path": str(model.model_path),
    }
    (OUT / "context.json").write_text(json.dumps(meta, indent=2))
    for p in OUT.glob("tabpfn_chunk_*.npy"):
        p.unlink()
    print(json.dumps(meta, indent=2))


def _context() -> int | None:
    return json.loads((OUT / "context.json").read_text())["context_arg"]


def lookup_quantiles(train: pd.DataFrame, test: pd.DataFrame, keys: list[list[str]],
                     min_n: int = 20) -> np.ndarray:
    """Empirical quantiles from the first key level whose training cell has >= min_n rows;
    the global distribution is the last fallback."""
    out = np.full((len(test), len(QS)), np.nan)
    tables = []
    for k in keys:
        g = train.groupby(k)["arr_delay"]
        tables.append((k, g.size(), g.quantile(QS).unstack()))
    glob = train["arr_delay"].quantile(QS).to_numpy()
    level = np.full(len(test), "global", dtype=object)
    for i, row in enumerate(test.itertuples(index=False)):
        for k, n, q in tables:
            key = tuple(getattr(row, c) for c in k)
            key = key[0] if len(key) == 1 else key
            if n.get(key, 0) >= min_n:
                out[i] = q.loc[key].to_numpy()
                level[i] = "x".join(k)
                break
        else:
            out[i] = glob
    return out, level


def run_baselines() -> None:
    import lightgbm as lgb

    train, test, _, _ = split(_context())
    meta = json.loads((OUT / "context.json").read_text())
    assert len(train) == meta["context_rows"] and len(test) == meta["test_rows"]
    preds = {"schedule (delay=0)": np.zeros((len(test), len(QS)))}
    preds["lookup origin"], lv1 = lookup_quantiles(train, test, [["origin"]])
    preds["lookup origin x carrier"], lv2 = lookup_quantiles(train, test, [["origin", "carrier"], ["origin"]])
    enc = Encoder(train)
    Xtr, Xte, ytr = enc.pandas_cat(train), enc.pandas_cat(test), train["arr_delay"].to_numpy()
    lg = np.zeros((len(test), len(QS)))
    for j, a in enumerate(QS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, num_threads=4, random_state=SEED, verbose=-1)
        m.fit(Xtr, ytr)
        lg[:, j] = m.predict(Xte)
    preds["LightGBM quantile"] = lg
    g = np.load(OUT / "tabpfn_grid.npy")
    preds["TabPFN-3.5"] = np.stack([g[:, np.argmin(np.abs(GRID - q))] for q in QS], axis=1)
    rows = []
    for name, p in preds.items():
        d = test[["origin", "carrier", "crs_arr_ts", "n_train_oc", "density", "arr_delay"]].copy()
        d["model"] = name
        for j, q in enumerate(QS):
            d[f"q{int(q*100)}"] = p[:, j]
        rows.append(d)
    pd.concat(rows).to_parquet(OUT / "preds.parquet")
    print(pd.Series(lv1).value_counts(), pd.Series(lv2).value_counts(), sep="\n")
    print(f"peak rss {_rss_gb():.2f} GB")


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def to_markdown(t: pd.DataFrame) -> str:
    def f(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def run_report() -> None:
    p = pd.read_parquet(OUT / "preds.parquet")
    p["bin"] = np.where(p["n_train_oc"] < 50, "<50 (incl. unseen)", p["density"])
    order = ["<50 (incl. unseen)", "50-199", "200-799", "800+", "all"]
    recs = []
    for (model, b), d in list(p.groupby(["model", "bin"])) + [((m, "all"), d) for m, d in p.groupby("model")]:
        y = d["arr_delay"].to_numpy()
        r = {"model": model, "bin": b, "n_test": len(d),
             "P10 cov": np.mean(y <= d["q10"]), "P90 cov": np.mean(y <= d["q90"])}
        for q in QS:
            r[f"pin{int(q*100)}"] = pinball(y, d[f"q{int(q*100)}"].to_numpy(), q)
        r["pin mean(10,50,90)"] = np.mean([r["pin10"], r["pin50"], r["pin90"]])
        recs.append(r)
    t = pd.DataFrame(recs)
    t["bin"] = pd.Categorical(t["bin"], order)
    models = list(dict.fromkeys(p["model"]))
    t["model"] = pd.Categorical(t["model"], models)
    t = t.sort_values(["bin", "model"])
    md = to_markdown(t)
    # Paired difference in mean pinball (10/50/90), TabPFN minus each other model, with a
    # 95% bootstrap CI that resamples whole arrival days (delays are correlated within a day).
    p["day"] = pd.to_datetime(p["crs_arr_ts"]).dt.date
    p["pin"] = sum(np.maximum(q * (p["arr_delay"] - p[f"q{int(q*100)}"]),
                              (q - 1) * (p["arr_delay"] - p[f"q{int(q*100)}"])) for q in QS) / len(QS)
    w = p.pivot_table(index=[p.index, "day", "bin"], columns="model", values="pin").reset_index()
    rng = np.random.default_rng(SEED)
    days = w["day"].unique()
    diff_lines = ["| other model | bin | TabPFN minus other (mean pinball) | 95% day-bootstrap CI |",
                  "|---|---|---|---|"]
    for b in order:
        wb = w if b == "all" else w[w["bin"] == b]
        for other in models:
            if other == "TabPFN-3.5":
                continue
            d = (wb["TabPFN-3.5"] - wb[other]).groupby(wb["day"]).agg(["sum", "size"])
            d = d.reindex(days, fill_value=0)
            boots = []
            for _ in range(2000):
                idx = rng.integers(0, len(d), len(d))
                s_, n_ = d["sum"].to_numpy()[idx].sum(), d["size"].to_numpy()[idx].sum()
                boots.append(s_ / n_ if n_ else np.nan)
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            diff_lines.append(f"| {other} | {b} | {d['sum'].sum() / d['size'].sum():+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    md = md + "\n\n" + "\n".join(diff_lines)
    (OUT / "results_table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tabpfn", "baselines", "report"])
    ap.add_argument("--context", type=int, default=None, help="TabPFN context rows (default: all)")
    a = ap.parse_args()
    {"tabpfn": lambda: run_tabpfn(a.context), "baselines": run_baselines, "report": run_report}[a.stage]()
