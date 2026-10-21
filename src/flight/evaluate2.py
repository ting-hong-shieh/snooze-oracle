"""Round 2: the round-1 comparison with and without regime features, holdout 2026-08.

Variants (same training rows, same test rows, same models, defaults, no tuning):
    without = round-1 features: origin, carrier, crs_arr_min_of_day, dow, month, distance
    with    = without + dest + origin_prev_dep_delay + dest_prev_arr_delay (see regime.py)
    dest only = without + dest (supplementary decomposition)
The lookups and the schedule-only rule cannot use the new features and are identical in both.

Stages (run from the repo root; outputs in data/flight2/):
    python -m src.flight.regime                            # airport-day table (CPU, DuckDB)
    python -m src.flight.evaluate2 tabpfn --variant with   # GPU; takes the lock per chunk
    python -m src.flight.evaluate2 tabpfn --variant without
    python -m src.flight.evaluate2 tabpfn --variant "dest only"
    python -m src.flight.evaluate2 baselines
    python -m src.flight.evaluate2 report                  # data/flight2/results_round2.md

The tabpfn stage takes /tmp/snooze-oracle-gpu.lock (fcntl.flock, the same lock as the
`flock` command) separately for each chunk of CHUNK test rows: fit + predict + free GPU
memory inside the lock, everything else outside. Finished chunks are cached as .npy files,
so a rerun skips them.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import resource
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .data import (CAT_FEATURES, CAT_FEATURES_REGIME, FEATURES, FEATURES_REGIME, SEED, Encoder,
                   split)
from .evaluate import GRID, QS, lookup_quantiles, to_markdown
from .regime import OUT2, add_regime

CHUNK = 500
LOCK = "/tmp/snooze-oracle-gpu.lock"
VARIANTS = {"without": (FEATURES, CAT_FEATURES), "with": (FEATURES_REGIME, CAT_FEATURES_REGIME),
            # Supplementary decomposition (added after seeing the with/without results):
            # round-1 features + dest only, no previous-day features.
            "dest only": (FEATURES + ["dest"], CAT_FEATURES + ["dest"])}
PAIRS = [("with", "without"), ("dest only", "without"), ("with", "dest only")]
BIN_ORDER = ["<50 (incl. unseen)", "50-199", "200-799", "800+", "all"]
N_BOOT = 2000


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _mem_status() -> dict:
    """Current anonymous / file-backed RSS and the high-water mark, GB (Linux /proc)."""
    out = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmHWM", "RssAnon", "RssFile"):
            out[k] = round(int(v.split()[0]) / 1e6, 2)
    return out


def load():
    """All flown training rows and all flown test rows, with regime features attached."""
    train, test, train_all, test_all = split(None)
    return add_regime(train), add_regime(test), add_regime(train_all), add_regime(test_all)


@contextmanager
def gpu_lock():
    with open(LOCK, "a") as fh:
        t_wait = time.time()
        fcntl.flock(fh, fcntl.LOCK_EX)
        t0 = time.time()
        try:
            yield t0 - t_wait
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def run_tabpfn(variant: str, chunks: list[int] | None) -> None:
    import torch
    from tabpfn import TabPFNRegressor

    feats, cats = VARIANTS[variant]
    d = OUT2 / f"tabpfn_{variant.replace(' ', '_')}"
    d.mkdir(parents=True, exist_ok=True)
    train, test, _, _ = load()
    enc = Encoder(train, feats, cats)
    Xtr, ytr, Xte = enc.numeric(train), train["arr_delay"].to_numpy(), enc.numeric(test)
    starts = list(range(0, len(test), CHUNK))
    log_path = d / "chunks.jsonl"
    print(f"{variant}: context {Xtr.shape}, test {len(test)}, chunks {len(starts)}", flush=True)
    for ci, i in enumerate(starts):
        if chunks is not None and ci not in chunks:
            continue
        path = d / f"chunk_{i:05d}.npy"
        if path.exists():
            continue
        with gpu_lock() as waited:
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            m = TabPFNRegressor(device="cuda", categorical_features_indices=enc.cat_indices,
                                random_state=SEED)
            m.fit(Xtr, ytr)
            q = m.predict(Xte[i:i + CHUNK], output_type="quantiles",
                          quantiles=[float(x) for x in GRID])
            rec = {"chunk": ci, "rows": [i, min(i + CHUNK, len(test))],
                   "gpu_s": None, "wait_s": round(waited, 1),
                   "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                   "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                   "n_estimators": getattr(m, "n_estimators_", m.n_estimators)}
            del m
            torch.cuda.empty_cache()
            rec["gpu_s"] = round(time.time() - t0, 1)
        np.save(path, np.stack(q, axis=1))
        rec["rss_gb"] = round(_rss_gb(), 2)
        rec.update(_mem_status())
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    done = [d / f"chunk_{i:05d}.npy" for i in starts]
    if all(p.exists() for p in done):
        np.save(d / "grid.npy", np.concatenate([np.load(p) for p in done]))
        print(f"{variant}: all chunks done -> {d / 'grid.npy'}")


def run_baselines() -> None:
    import lightgbm as lgb

    train, test, _, _ = load()
    keep = ["origin", "carrier", "dest", "crs_arr_ts", "n_train_oc", "density", "arr_delay",
            "origin_prev_dep_delay", "dest_prev_arr_delay"]
    rows = []

    def add(name, variant, p):
        d = test[keep].copy()
        d["model"], d["variant"] = name, variant
        for j, q in enumerate(QS):
            d[f"q{int(q * 100)}"] = p[:, j]
        rows.append(d)

    sched = np.zeros((len(test), len(QS)))
    lo, _ = lookup_quantiles(train, test, [["origin"]])
    loc, _ = lookup_quantiles(train, test, [["origin", "carrier"], ["origin"]])
    for v, (feats, cats) in VARIANTS.items():
        add("schedule (delay=0)", v, sched)
        add("lookup origin", v, lo)
        add("lookup origin x carrier", v, loc)
        enc = Encoder(train, feats, cats)
        Xtr, Xte, ytr = enc.pandas_cat(train), enc.pandas_cat(test), train["arr_delay"].to_numpy()
        lg = np.zeros((len(test), len(QS)))
        for j, a in enumerate(QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=a, num_threads=4,
                                  random_state=SEED, verbose=-1)
            m.fit(Xtr, ytr)
            lg[:, j] = m.predict(Xte)
        add("LightGBM quantile", v, lg)
        g = np.load(OUT2 / f"tabpfn_{v.replace(' ', '_')}" / "grid.npy")
        assert len(g) == len(test)
        add("TabPFN-3.5", v, np.stack([g[:, np.argmin(np.abs(GRID - q))] for q in QS], axis=1))
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT2 / "preds2.parquet")
    print(f"rows {len(out)}, peak rss {_rss_gb():.2f} GB")


# ---------------------------------------------------------------- report

METRICS = ["P10 cov", "P90 cov", "pin10", "pin50", "pin90", "pin mean"]


def _row_metrics(d: pd.DataFrame) -> np.ndarray:
    """Per-row contributions whose mean is each metric (columns in METRICS order)."""
    y = d["arr_delay"].to_numpy()
    cols = [(y <= d["q10"].to_numpy()).astype(float), (y <= d["q90"].to_numpy()).astype(float)]
    pins = []
    for q in QS:
        e = y - d[f"q{int(q * 100)}"].to_numpy()
        pins.append(np.maximum(q * e, (q - 1) * e))
    return np.column_stack(cols + pins + [np.mean(pins, axis=0)])


class DayBoot:
    """Day-block bootstrap: resample whole scheduled-arrival dates with replacement; a
    statistic is sum over drawn days / count over drawn days."""

    def __init__(self, days: np.ndarray):
        self.days = np.array(sorted(set(days)))
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(self.days), (N_BOOT, len(self.days)))
        self.w = np.zeros((N_BOOT, len(self.days)))
        for b in range(N_BOOT):
            np.add.at(self.w[b], idx[b], 1.0)

    def ci(self, days: np.ndarray, vals: np.ndarray):
        """vals: rows x k. Returns point (k,), lo (k,), hi (k,)."""
        pos = np.searchsorted(self.days, days)
        s = np.zeros((len(self.days), vals.shape[1]))
        np.add.at(s, pos, vals)
        n = np.bincount(pos, minlength=len(self.days)).astype(float)
        bs, bn = self.w @ s, self.w @ n
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = bs / bn[:, None]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        return s.sum(0) / n.sum(), lo, hi


def _fmt(p, lo, hi, signed=False):
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def run_report() -> None:
    p = pd.read_parquet(OUT2 / "preds2.parquet")
    p["bin"] = np.where(p["n_train_oc"] < 50, "<50 (incl. unseen)", p["density"])
    p["day"] = pd.to_datetime(p["crs_arr_ts"]).dt.normalize().to_numpy()
    boot = DayBoot(p["day"].to_numpy())
    models = list(dict.fromkeys(p["model"]))
    md = []
    for v in VARIANTS:
        md.append(f"### Variant `{v}`\n")
        recs = []
        for b in BIN_ORDER:
            for m in models:
                d = p[(p.variant == v) & (p.model == m) & ((p.bin == b) | (b == "all"))]
                pt, lo, hi = boot.ci(d["day"].to_numpy(), _row_metrics(d))
                r = {"model": m, "bin": b, "n_test": len(d)}
                r.update({k: _fmt(pt[i], lo[i], hi[i]) for i, k in enumerate(METRICS)})
                recs.append(r)
        md.append(to_markdown(pd.DataFrame(recs)) + "\n")
    # Paired with minus without, same rows.
    for va, vz in PAIRS:
        md.append(f"### Paired difference: `{va}` minus `{vz}` (same test rows)\n")
        recs = []
        for m in ["TabPFN-3.5", "LightGBM quantile"]:
            for b in BIN_ORDER:
                sel = (p.model == m) & ((p.bin == b) | (b == "all"))
                a, z = p[sel & (p.variant == va)], p[sel & (p.variant == vz)]
                assert (a["arr_delay"].to_numpy() == z["arr_delay"].to_numpy()).all()
                diff = _row_metrics(a) - _row_metrics(z)
                pt, lo, hi = boot.ci(a["day"].to_numpy(), diff)
                r = {"model": m, "bin": b, "n_test": len(a)}
                r.update({k: _fmt(pt[i], lo[i], hi[i], signed=True) for i, k in enumerate(METRICS)})
                recs.append(r)
        md.append(to_markdown(pd.DataFrame(recs)) + "\n")
    md.append(regime_analysis(p, boot))
    md.append(artefact_check(p, boot))
    text = "\n".join(md)
    (OUT2 / "results_round2.md").write_text(text)
    print(text)


def regime_analysis(p: pd.DataFrame, boot: DayBoot) -> str:
    """Coverage on days after bad vs calm days, with thresholds fixed on training days."""
    from .regime import _table

    t = _table()
    t = t[(t.kind == "dest_arr") & (t.n_flights > 0)]
    tr_days = t[(pd.to_datetime(t.day) >= "2025-08-01") & (pd.to_datetime(t.day) < "2026-08-01")]
    q75 = float(np.nanquantile(tr_days["mean_delay"], 0.75))
    q25 = float(np.nanquantile(tr_days["mean_delay"], 0.25))
    out = [f"### P90 coverage by regime of day A-2 at the destination\n",
           f"Groups by `dest_prev_arr_delay` (mean ArrDelay at JFK/LGA two days before arrival); "
           f"thresholds = quartiles of that daily mean over training-period JFK/LGA days: "
           f"calm < {q25:.1f} min, bad >= {q75:.1f} min.\n"]
    p = p.copy()
    x = p["dest_prev_arr_delay"]
    p["regime"] = np.where(x >= q75, "bad (top quartile)",
                           np.where(x < q25, "calm (bottom quartile)", "middle"))
    recs = []
    for g in ["calm (bottom quartile)", "middle", "bad (top quartile)"]:
        base = p[(p.regime == g) & (p.model == "schedule (delay=0)") & (p.variant == "without")]
        y = base["arr_delay"]
        for m in ["lookup origin x carrier", "LightGBM quantile", "TabPFN-3.5"]:
            for v in VARIANTS:
                if m.startswith("lookup") and v == "with":
                    continue
                d = p[(p.regime == g) & (p.model == m) & (p.variant == v)]
                pt, lo, hi = boot.ci(d["day"].to_numpy(), _row_metrics(d)[:, [1]])
                recs.append({"A-2 regime at dest": g, "n_test": len(d),
                             "test days": d["day"].nunique(),
                             "observed P90 ArrDelay": round(float(y.quantile(0.9)), 1),
                             "model": m if not m.startswith("lookup") else m + " (no regime)",
                             "variant": v, "P90 cov": _fmt(pt[0], lo[0], hi[0]),
                             "median predicted P90": round(float(d["q90"].median()), 1)})
    out.append(to_markdown(pd.DataFrame(recs)) + "\n")
    # Rank correlations, flown rows.
    one = p[(p.model == "schedule (delay=0)") & (p.variant == "without")]
    train, _, _, _ = load()
    lines = ["| feature | Spearman with ArrDelay, train | Spearman with ArrDelay, test 2026-08 | "
             "missing train / test |", "|---|---|---|---|"]
    for f in ["origin_prev_dep_delay", "dest_prev_arr_delay"]:
        lines.append(f"| {f} | {train[f].corr(train['arr_delay'], method='spearman'):+.3f} | "
                     f"{one[f].corr(one['arr_delay'], method='spearman'):+.3f} | "
                     f"{int(train[f].isna().sum())} / {int(one[f].isna().sum())} |")
    out.append("\n".join(lines) + "\n")
    # Persistence of the daily JFK/LGA mean arrival delay.
    a = t.pivot(index="day", columns="airport", values="mean_delay").sort_index()
    lines = ["| airport | autocorr lag 1 day | lag 2 days | lag 3 days |", "|---|---|---|---|"]
    for ap in a:
        s = a[ap]
        lines.append(f"| {ap} | {s.autocorr(1):.3f} | {s.autocorr(2):.3f} | {s.autocorr(3):.3f} |")
    out.append("Daily mean ArrDelay at JFK/LGA, 2025-07 ... 2026-08:\n\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def artefact_check(p: pd.DataFrame, boot: DayBoot, cut: float = 150.0) -> str:
    """Post-hoc: predicted P90 above `cut` minutes, and the with-vs-without P90 coverage
    change with the test days that have any such TabPFN `with` prediction left out."""
    lines = [f"### Post-hoc check: predicted P90 above {cut:.0f} min\n",
             "| model | variant | rows with q90 > cut | days |", "|---|---|---|---|"]
    for (m, v), d in p[p.model.isin(["TabPFN-3.5", "LightGBM quantile"])].groupby(["model", "variant"]):
        big = d[d.q90 > cut]
        days = ", ".join(sorted({str(x)[:10] for x in big["day"]}))
        lines.append(f"| {m} | {v} | {len(big)} | {days} |")
    w = p[(p.model == "TabPFN-3.5") & (p.variant == "with")]
    bad_days = set(w.loc[w.q90 > cut, "day"])
    lines += ["", "P90 coverage change, all test days vs without those days:", "",
              "| model | pair | all days | without flagged days |", "|---|---|---|---|"]
    for m in ["TabPFN-3.5", "LightGBM quantile"]:
        for va, vz in PAIRS:
            cells = []
            for keep in (None, bad_days):
                sel = (p.model == m) & (~p["day"].isin(keep) if keep else True)
                a, z = p[sel & (p.variant == va)], p[sel & (p.variant == vz)]
                diff = _row_metrics(a)[:, [1]] - _row_metrics(z)[:, [1]]
                pt, lo, hi = boot.ci(a["day"].to_numpy(), diff)
                cells.append(_fmt(pt[0], lo[0], hi[0], signed=True))
            lines.append(f"| {m} | {va} - {vz} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tabpfn", "baselines", "report"])
    ap.add_argument("--variant", choices=list(VARIANTS), default="with")  # "dest only" needs quotes
    ap.add_argument("--chunks", type=str, default=None, help="comma-separated chunk indices")
    a = ap.parse_args()
    ch = None if a.chunks is None else [int(c) for c in a.chunks.split(",")]
    {"tabpfn": lambda: run_tabpfn(a.variant, ch), "baselines": run_baselines,
     "report": run_report}[a.stage]()
