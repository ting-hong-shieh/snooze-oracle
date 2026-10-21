"""Commute scenario: wake up, get ready, take an Uber to a meeting (no flight).

Same alarm logic as the pickup scenario (core.Scenario, Monte Carlo over prep and ride),
with a fixed deadline: on time  <=>  wake + prep + ride <= meeting.

Usage (from the repo root):
    python -m src.alarm.commute commute-routes   # data/commute/routes.csv
    python -m src.alarm.commute tables            # ride tables (TabPFN takes the GPU lock)
    python -m src.alarm.commute run               # backtest rows (CPU)
    python -m src.alarm.commute report            # docs/commute_backtest.md + charts

Setting: weekdays (Mon-Fri), request time 05:00-10:59, drop-off in one of the 6 destination
zones. Train = July 2026, test = August 2026 (strict split on request time).
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Boot, ci
from .core import THRESHOLD, Scenario, _ride_table
from .tables import LEVELS, OUT, build_ride

ROOT = OUT.parents[1]
OUTC = ROOT / "data" / "commute"
ROUTES = OUTC / "routes.csv"
DOCS = ROOT / "docs"
AIRPORTS = (1, 132, 138)
UNKNOWN = (264, 265)
HOURS = range(5, 11)                                # request 05:00-10:59
GRID = np.r_[np.arange(300, 651, 10), 659]          # ride table request minutes
MEETINGS = (480, 510, 540, 570, 600)                # 08:00 ... 10:00
GUT_B = (0, 5, 10, 15, 20, 30)
GUT_CURVE = tuple(range(0, 31))                     # reference curve, every minute
REF_B = 10
PREP_MIN, PREP_SD = 15.0, 5.0
REALITY_DRAWS = 5
RADII = (15, 30)
WAKE_RANGE = (-240, 0)
SEED = 20261022
MODELS = ("lookup", "lgbm", "tabpfn")
PREFIX = "commute_ride"


# ------------------------------------------------------------------ data and routes

def _sql(month_start: str, month_end: str, extra: str = "") -> str:
    from src.ride.retrieval import COLS, PROCESSED

    path = PROCESSED / f"uber_trips_{month_start[:7]}.parquet"
    return f"""
        SELECT {COLS} FROM read_parquet('{path}')
        WHERE dow < 5
          AND request_ts >= TIMESTAMP '{month_start}' AND request_ts < TIMESTAMP '{month_end}'
          {extra}
    """


@lru_cache(maxsize=None)
def destinations() -> tuple[int, ...]:
    """The 6 zones with the most July weekday drop-offs requested 07:00-09:59 (airports and
    unknown zones excluded)."""
    from src.ride.retrieval import connect

    d = connect().execute(f"""
        SELECT do_zone, count(*) AS n FROM ({_sql('2026-07-01', '2026-08-01')})
        WHERE req_hour BETWEEN 7 AND 9 AND do_zone NOT IN {AIRPORTS + UNKNOWN}
        GROUP BY do_zone ORDER BY n DESC, do_zone LIMIT 6
    """).fetchdf()
    return tuple(int(z) for z in d["do_zone"])


@lru_cache(maxsize=None)
def load(month: str) -> pd.DataFrame:
    """Weekday trips requested 05:00-10:59 to the destination zones, in July or August."""
    from src.ride.retrieval import connect

    start, end = {"2026-07": ("2026-07-01", "2026-08-01"),
                  "2026-08": ("2026-08-01", "2026-09-01")}[month]
    path = OUTC / f"trips_{month}.parquet"
    if not path.exists():
        OUTC.mkdir(parents=True, exist_ok=True)
        extra = (f"AND req_min_of_day BETWEEN {GRID[0]} AND {GRID[-1]} "
                 f"AND do_zone IN {destinations()}")
        connect().execute(f"COPY ({_sql(start, end, extra)} ORDER BY request_ts, pu_zone) "
                          f"TO '{path}' (FORMAT parquet)")
    return pd.read_parquet(path)


def choose_routes() -> pd.DataFrame:
    """12 origin -> destination routes, counted on weekday trips requested 06:00-09:59:
    6 dense = the most July trips; 6 sparse = 10-29 July trips, one per borough in turn
    (boroughs ordered by their best candidate), each borough's candidates ranked by August
    trips (so the same-date reality pool is as full as possible), then by zone id.
    Origins exclude the destination zones, airports and unknown zones."""
    from src.ride.retrieval import zone_borough

    dest = destinations()
    excl = set(dest) | set(AIRPORTS) | set(UNKNOWN)

    def counts(month):
        t = load(month)
        t = t[t["req_hour"].between(6, 9) & ~t["pu_zone"].isin(excl)]
        return t.groupby(["pu_zone", "do_zone"]).size()

    d = pd.concat([counts("2026-07").rename("jul"), counts("2026-08").rename("aug")],
                  axis=1).fillna(0).astype(int).reset_index()
    d["borough"] = d["pu_zone"].map(zone_borough())
    dense = d.sort_values(["jul", "pu_zone"], ascending=[False, True]).head(6)
    cand = d[d["jul"].between(10, 29) & ~d["borough"].isin(["EWR", "Unknown", "N/A"])]
    cand = cand.sort_values(["aug", "jul", "pu_zone"], ascending=[False, False, True])
    queues = {b: list(g.index) for b, g in cand.groupby("borough", sort=False)}
    order = list(dict.fromkeys(cand["borough"]))       # boroughs by their best candidate
    picked = []
    while len(picked) < 6 and any(queues.values()):
        for b in order:
            if queues[b] and len(picked) < 6:
                picked.append(queues[b].pop(0))
    r = pd.concat([dense.assign(kind="dense"), cand.loc[picked].assign(kind="sparse")])
    names = pd.read_csv(ROOT / "data" / "raw" / "taxi_zone_lookup.csv").set_index("LocationID")
    r["pu_name"] = r["pu_zone"].map(names["Zone"])
    r["do_name"] = r["do_zone"].map(names["Zone"])
    r["do_borough"] = r["do_zone"].map(zone_borough())
    r = r.reset_index(drop=True)
    r.insert(0, "route_id", [f"{a}->{b}" for a, b in zip(r.pu_zone, r.do_zone)])
    r.to_csv(ROUTES, index=False)
    print(f"destinations: {dest}")
    print(r.to_string(index=False))
    return r


def weekdays() -> list[pd.Timestamp]:
    return [d for d in pd.date_range("2026-08-01", "2026-08-31") if d.dayofweek < 5]


def tables(models=MODELS) -> None:
    r = pd.read_csv(ROUTES)
    build_ride(list(zip(r.pu_zone, r.do_zone)), weekdays(), models, train=load("2026-07"),
               grid=GRID, hours=HOURS, prefix=PREFIX)


# ------------------------------------------------------------------ alarm

class CommuteScenario(Scenario):
    """Fixed deadline (the meeting); no flight. Reuses Scenario.ride / p_on_time /
    latest_safe_wake with the commute ride table and request-time grid."""

    grid = GRID
    wake_range = WAKE_RANGE

    def __init__(self, pu_zone: int, do_zone: int, date, meeting_min: float,
                 model: str = "tabpfn", prep_min: float = PREP_MIN, prep_sd: float = PREP_SD,
                 n: int = 2000, seed: int = 0):
        self.ride_q = _ride_table(model, PREFIX, tuple(GRID))[(int(pu_zone), int(do_zone),
                                                              pd.Timestamp(date).date())]
        rng = np.random.default_rng(seed)
        u_r = rng.uniform(LEVELS[0], LEVELS[-1], n)
        self.prep = np.maximum(rng.normal(prep_min, prep_sd, n), 0.0)
        pos = np.interp(u_r, LEVELS, np.arange(len(LEVELS)))
        self.li = np.minimum(pos.astype(int), len(LEVELS) - 2)
        self.lw = pos - self.li
        self.date = pd.Timestamp(date).normalize()
        self.crs = float(meeting_min)
        self.ready = np.full(n, float(meeting_min))


# ------------------------------------------------------------------ backtest

class SameDayPool:
    """Real August trips per (route, date), for the reality ride draws."""

    def __init__(self):
        t = load("2026-08")
        t = t.assign(date=t["request_ts"].dt.normalize())
        self.g = {k: (v["req_min_of_day"].to_numpy(), v["total_min"].to_numpy())
                  for k, v in t.groupby(["pu_zone", "do_zone", "date"])}

    def draw(self, key, request_min: np.ndarray, u: np.ndarray):
        m, v = self.g.get(key, (np.array([]), np.array([])))
        out, rad = np.full(len(u), np.nan), np.zeros(len(u), int)
        for i, (t, uu) in enumerate(zip(request_min, u)):
            for r in RADII:
                w = np.sort(v[np.abs(m - t) <= r])
                if len(w):
                    out[i], rad[i] = np.quantile(w, uu), r
                    break
        return out, rad


def lookup_median(table: dict, key, minute: float) -> float:
    q = table[key]
    t = int(np.argmin(np.abs(GRID - np.clip(minute, GRID[0], GRID[-1]))))
    return float(np.interp(0.5, LEVELS, q[t]))


def run() -> None:
    t0 = time.time()
    routes = pd.read_csv(ROUTES)
    lk = _ride_table("lookup", PREFIX, tuple(GRID))
    pool = SameDayPool()
    rows, skipped, infeasible, total = [], {"dense": 0, "sparse": 0}, {m: 0 for m in MODELS}, 0
    for date in weekdays():
        for ri, rt in routes.iterrows():
            key = (int(rt.pu_zone), int(rt.do_zone), date.date())
            for meet in MEETINGS:
                total += 1
                sid = f"{date.date()}_{rt.route_id}_{meet}"
                med = lookup_median(lk, key, meet - 30)
                wake = {f"gut_{b}": meet - PREP_MIN - med - b for b in GUT_CURVE}
                for m in MODELS:
                    s = CommuteScenario(rt.pu_zone, rt.do_zone, date, meet, m,
                                        seed=SEED + total)
                    w, ok = s.latest_safe_wake()
                    infeasible[m] += not ok
                    wake[m] = w
                rng = np.random.default_rng([SEED, date.day, ri, meet])
                prep = np.maximum(rng.normal(PREP_MIN, PREP_SD, REALITY_DRAWS), 0.0)
                u = rng.uniform(0, 1, REALITY_DRAWS)
                res = {}
                for k, w in wake.items():
                    res[k] = (w, *pool.draw((key[0], key[1], date), w + prep, u))
                if any(np.isnan(r[1]).any() for r in res.values()):
                    skipped[rt.kind] += 1
                    continue
                for k, (w, ride, rad) in res.items():
                    arrive = w + prep + ride
                    for j in range(REALITY_DRAWS):
                        rows.append((sid, str(date.date()), rt.kind, rt.route_id, meet, k, w,
                                     bool(arrive[j] > meet), float(max(0.0, meet - arrive[j])),
                                     j, int(rad[j])))
    out = pd.DataFrame(rows, columns=["scenario", "date", "kind", "route_id", "meeting",
                                      "strategy", "wake_min", "late", "early_min", "draw",
                                      "radius"])
    out.to_parquet(OUTC / "backtest_rows.parquet", index=False)
    kept = out.drop_duplicates("scenario")
    meta = {"weekdays": len(weekdays()), "scenarios": total, "skipped": skipped,
            "kept": {k: int(v) for k, v in kept["kind"].value_counts().items()},
            "infeasible": infeasible,
            "draws_at_30min": float((out["radius"] == 30).mean()),
            "seconds": round(time.time() - t0)}
    pd.Series(meta).to_json(OUTC / "backtest_meta.json")
    print(meta)


# ------------------------------------------------------------------ report

STRATS = [f"gut_{b}" for b in GUT_B] + list(MODELS)
LABEL = {**{f"gut_{b}": f"gut B={b}" for b in GUT_CURVE}, "lookup": "lookup p>=0.90",
         "lgbm": "LightGBM p>=0.90", "tabpfn": "TabPFN p>=0.90"}
COLOR = {"lookup": "#1baf7a", "lgbm": "#eda100", "tabpfn": "#2a78d6"}
MARKER = {"lookup": "^", "lgbm": "D", "tabpfn": "o"}
INK, INK2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def curve_gap(st: dict, s: str) -> np.ndarray:
    gw = np.stack([st[f"gut_{b}"]["early"] for b in GUT_CURVE])
    gl = np.stack([st[f"gut_{b}"]["late"] for b in GUT_CURVE])
    out = np.full(gw.shape[1], np.nan)
    for r in range(gw.shape[1]):
        o = np.argsort(gw[:, r], kind="stable")
        x = st[s]["early"][r]
        if gw[o[0], r] <= x <= gw[o[-1], r]:
            out[r] = st[s]["late"][r] - np.interp(x, gw[o, r], gl[o, r])
    return out


def report() -> None:
    rows = pd.read_parquet(OUTC / "backtest_rows.parquet")
    meta = pd.read_json(OUTC / "backtest_meta.json", typ="series")
    routes = pd.read_csv(ROUTES)
    boot = Boot(pd.to_datetime(rows["date"]).dt.day.to_numpy())
    ref = rows[rows.strategy == f"gut_{REF_B}"].set_index(["scenario", "draw"])["wake_min"]
    rows["wake_vs_ref"] = rows["wake_min"].to_numpy() - ref.loc[
        list(zip(rows.scenario, rows.draw))].to_numpy()
    late, early, wk = (rows[c].to_numpy(float) for c in ("late", "early_min", "wake_vs_ref"))
    strat = rows["strategy"].to_numpy()

    md = ["# Commute backtest, August 2026 weekdays", "",
          "Wake up, get ready (prep ~ N(15, 5) min), request an Uber, arrive before the meeting. "
          "Strategies pick one wake time the night before; reality uses real trips on the same "
          "date and route. Method, judgment calls and fairness notes are at the end.", "",
          "## Routes", "",
          "Trips counted on weekday requests 06:00-09:59 (July = training month).", "",
          "| kind | route | origin | borough | destination | July trips | August trips |",
          "|---|---|---|---|---|---|---|"]
    for r in routes.itertuples():
        md.append(f"| {r.kind} | {r.route_id} | {r.pu_name} | {r.borough} | {r.do_name} | "
                  f"{r.jul} | {r.aug} |")
    md += ["", f"Destinations (most July weekday drop-offs requested 07:00-09:59, airports "
               f"excluded): {', '.join(map(str, destinations()))}.", "",
           "## Scenarios", "",
           f"{meta.weekdays} August weekdays x 12 routes x 5 meeting times (08:00-10:00) = "
           f"{meta.scenarios} scenarios. Skipped because some strategy's request time had no "
           f"real trip on that date and route within +-30 min: {meta.skipped}. Kept: "
           f"{meta.kept}. Reality draws that needed the +-30 min window: "
           f"{meta.draws_at_30min:.1%}. Wake ranges with no time reaching 0.90: "
           f"{meta.infeasible}. {REALITY_DRAWS} reality draws per scenario.", "",
           "95% intervals: day-level bootstrap (2,000 resamples of the August weekdays).", ""]

    all_stats = {}
    for split in ("all", "dense", "sparse"):
        base = np.ones(len(rows), bool) if split == "all" else (rows["kind"] == split).to_numpy()
        if not base.any():
            continue
        st = {}
        for s in [f"gut_{b}" for b in GUT_CURVE] + list(MODELS):
            m = base & (strat == s)
            st[s] = {"late": boot.mean(late, m), "early": boot.median(early, m),
                     "wake": boot.mean(wk, m), "n": int(m.sum())}
        all_stats[split] = st
        n_sc = rows.loc[base, "scenario"].nunique()
        md += [f"## {split} routes ({n_sc:,} scenarios)", "",
               f"| strategy | late rate | median minutes early | mean wake vs gut B={REF_B} (min) |",
               "|---|---|---|---|"]
        for s in STRATS:
            md.append(f"| {LABEL[s]} | {ci(st[s]['late'])} | {ci(st[s]['early'], 1)} | "
                      f"{ci(st[s]['wake'], 1)} |")
        md += ["", "Gap to the gut-rule curve (late rate minus the curve's late rate at the same "
                   "median minutes early; curve = gut rules for every B from 0 to 30 min, linear "
                   "between them; negative = below the curve):", ""]
        for s in MODELS:
            g = curve_gap(st, s)
            ok = ~np.isnan(g[1:])
            md.append(f"- {LABEL[s]}: " + (ci(g) if not np.isnan(g[0]) and ok.mean() > 0.95 else
                      f"outside the curve's range in {1 - ok.mean():.0%} of replicates "
                      f"(point estimate {'inside' if not np.isnan(g[0]) else 'outside'})"))
        md += ["", "Paired differences (same scenarios and draws):", ""]
        for o in ("lookup", "lgbm"):
            dl = st["tabpfn"]["late"] - st[o]["late"]
            de = st["tabpfn"]["early"] - st[o]["early"]
            dw = st["tabpfn"]["wake"] - st[o]["wake"]
            md.append(f"- TabPFN minus {LABEL[o]}: late rate {ci(dl)}, median minutes early "
                      f"{ci(de, 1)}, wake time {ci(dw, 1)} min")
        md.append("")

    md += risk_section(rows, boot, late, strat, routes, all_stats["all"])
    md += [NOTES]
    (DOCS / "commute_backtest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    frontier_chart(all_stats)


def risk_section(rows, boot, late, strat, routes, st_all) -> list[str]:
    rid = rows["route_id"].to_numpy()
    order = list(routes["route_id"])
    n_sc = rows.drop_duplicates("scenario").groupby("route_id").size().reindex(order).fillna(0)
    per = {}
    for s in [f"gut_{b}" for b in GUT_CURVE] + list(MODELS):
        per[s] = np.stack([boot.mean(late, (rid == r) & (strat == s)) if n_sc[r] else
                           np.full(len(boot.counts) + 1, np.nan) for r in order])
    # the fixed buffer whose overall late rate is closest to 10%
    comp = min(GUT_CURVE, key=lambda b: abs(st_all[f"gut_{b}"]["late"][0] - 0.10))
    md = ["## Risk by route", "",
          "Late rate per route, 95% day-bootstrap intervals.", "",
          "| route | kind | scenarios | " + " | ".join(LABEL[s] for s in STRATS) + " |",
          "|---|---|---|" + "---|" * len(STRATS)]
    kind = dict(zip(routes.route_id, routes.kind))
    for i, r in enumerate(order):
        cells = []
        for s in STRATS:
            a = per[s][i]
            if np.isnan(a[0]):
                cells.append("-")
                continue
            lo, hi = np.nanpercentile(a[1:], [2.5, 97.5])
            cells.append(f"{100 * a[0]:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]")
        md.append(f"| {r} | {kind[r]} | {int(n_sc[r])} | " + " | ".join(cells) + " |")
    # Weights per route and bootstrap replicate = kept scenarios of that route on the drawn
    # days (a route with no scenario on the drawn days gets weight 0 in that replicate).
    sc = rows[(rows["draw"] == 0) & (rows["strategy"] == f"gut_{REF_B}")]
    day = boot.idx[sc.index.to_numpy()]
    N = np.zeros((len(order), len(boot.days)))
    np.add.at(N, (pd.Index(order).get_indexer(sc["route_id"]), day), 1)
    W = np.column_stack([N.sum(1), N @ boot.counts.T])          # (routes, 1 + N_BOOT)

    def spread(a):
        a = np.where(W > 0, np.nan_to_num(a), 0.0)
        mu = (W * a).sum(0) / W.sum(0)
        sd = np.sqrt((W * (a - mu) ** 2).sum(0) / W.sum(0))
        mad = (W * np.abs(a - 0.10)).sum(0) / W.sum(0)
        return sd, mad

    md += ["", "Spread across routes (weights = kept scenarios per route). MAD10 = weighted mean "
               "absolute distance of the route late rates from 10%, in points. The fixed buffer "
               f"whose overall late rate is closest to 10% is gut B={comp}.", "",
           "| strategy | min | max | weighted SD [95% CI] | MAD10 [95% CI] |", "|---|---|---|---|---|"]
    res = {}
    for s in STRATS + ([f"gut_{comp}"] if comp not in GUT_B else []):
        a = per[s]
        sd, mad = spread(a)
        res[s] = (sd, mad)
        i0, i1 = int(np.nanargmin(a[:, 0])), int(np.nanargmax(a[:, 0]))
        md.append(f"| {LABEL[s]} | {100 * a[i0, 0]:.0f}% ({order[i0]}) | "
                  f"{100 * a[i1, 0]:.0f}% ({order[i1]}) | {ci(100 * sd, 1)} | {ci(100 * mad, 1)} |")
    md += ["", f"Model strategies minus gut B={comp} (paired over the same bootstrap draws):", ""]
    for s in MODELS:
        dsd = 100 * (res[s][0] - res[f"gut_{comp}"][0])
        dmad = 100 * (res[s][1] - res[f"gut_{comp}"][1])
        md.append(f"- {LABEL[s]}: weighted SD {ci(dsd, 1)}, MAD10 {ci(dmad, 1)} points")
    md.append("")
    risk_chart(order, kind, n_sc, per, comp)
    return md


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDC, lw=1)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)


def frontier_chart(all_stats: dict) -> None:
    """Top row: the whole gut-rule curve. Bottom row: zoom around the model strategies,
    where they are labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = list(all_stats)
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 9.5), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1, 1.1]}, squeeze=False)
    offset = {"lookup": (10, 8), "lgbm": (10, -14), "tabpfn": (-10, -16)}
    for col, split in enumerate(splits):
        st = all_stats[split]
        ex = [st[s]["early"][0] for s in MODELS]
        ly = [100 * st[s]["late"][0] for s in MODELS]
        for row, ax in enumerate(axes[:, col]):
            _style(ax)
            ax.plot([st[f"gut_{b}"]["early"][0] for b in GUT_CURVE],
                    [100 * st[f"gut_{b}"]["late"][0] for b in GUT_CURVE], color=INK2, lw=2,
                    zorder=2, label="gut rules, B = 0 ... 30 min")
            for b in GUT_B:
                x, y = st[f"gut_{b}"]["early"][0], 100 * st[f"gut_{b}"]["late"][0]
                ax.plot(x, y, marker="o", ms=8, mfc=SURFACE, mec=INK2, mew=2, zorder=2)
                ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, 4),
                            fontsize=9, color=INK2, annotation_clip=True)
            for s in MODELS:
                e, l = st[s]["early"], 100 * st[s]["late"]
                ax.errorbar(e[0], l[0],
                            xerr=[[e[0] - np.percentile(e[1:], 2.5)],
                                  [np.percentile(e[1:], 97.5) - e[0]]],
                            yerr=[[l[0] - np.percentile(l[1:], 2.5)],
                                  [np.percentile(l[1:], 97.5) - l[0]]],
                            fmt=MARKER[s], ms=9, color=COLOR[s], mec=SURFACE, mew=1.5,
                            ecolor=COLOR[s], elinewidth=1.5, capsize=0, zorder=3, label=LABEL[s])
                if row == 1:
                    ax.annotate(LABEL[s], (e[0], l[0]), textcoords="offset points",
                                xytext=offset[s], fontsize=9, color=INK,
                                ha="right" if offset[s][0] < 0 else "left")
            ax.set_xlabel("median minutes early", color=INK2)
            if row == 0:
                ax.set_title(f"{split} routes", color=INK, fontsize=12, loc="left")
            else:
                ax.set_xlim(min(ex) - 5, max(ex) + 5)
                ax.set_ylim(0, max(25.0, max(ly) + 10))
                ax.set_title("zoom: around the model strategies", color=INK2, fontsize=10,
                             loc="left")
        axes[0, col].set_ylabel("late rate (%)" if col == 0 else "", color=INK2)
        axes[1, col].set_ylabel("late rate (%)" if col == 0 else "", color=INK2)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Commuting by Uber to a meeting, August 2026 weekdays: late rate vs minutes "
                 "early (95% day-bootstrap intervals)", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(DOCS / "commute_frontier.png", dpi=150, facecolor=SURFACE)
    print("wrote docs/commute_frontier.png")


def risk_chart(order, kind, n_sc, per, comp) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [f"gut_{REF_B}"] + ([f"gut_{comp}"] if comp != REF_B else []) + list(MODELS)
    color = {**COLOR, f"gut_{REF_B}": INK2, f"gut_{comp}": "#8a8984"}
    marker = {**MARKER, f"gut_{REF_B}": "o", f"gut_{comp}": "s"}
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", visible=False)
    offs = np.linspace(-0.3, 0.3, len(shown))
    for o, s in zip(offs, shown):
        a = 100 * per[s]
        lo, hi = np.nanpercentile(a[:, 1:], [2.5, 97.5], axis=1)
        hollow = s.startswith("gut")
        ax.errorbar(x + o, a[:, 0], yerr=[a[:, 0] - lo, hi - a[:, 0]], fmt=marker[s], ms=8,
                    color=color[s], mfc=SURFACE if hollow else color[s],
                    mec=color[s] if hollow else SURFACE, mew=1.8, ecolor=color[s],
                    elinewidth=1.4, capsize=0, label=LABEL[s], zorder=3)
    ax.axhline(10, color=INK, lw=1, ls="--", zorder=2)
    ax.annotate("10% target", (-0.62, 10), xytext=(0, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, color=INK)
    ax.set_xlim(-0.65, len(order) - 0.4)
    for i in range(len(order) - 1):
        if kind[order[i]] != kind[order[i + 1]]:
            ax.axvline(i + 0.5, color=GRIDC, lw=2)
    ax.set_xticks(x, [f"{r}\n{kind[r]}, {int(n_sc[r])}" for r in order], fontsize=8, color=INK2)
    ax.set_ylabel("late rate (%)", color=INK2)
    ax.set_xlabel("route (pickup zone -> destination zone; kind, kept scenarios)", color=INK2)
    ax.legend(ncol=len(shown), loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Commute late rate by route, August 2026 weekdays (95% day-bootstrap "
                 "intervals)", color=INK, fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / "commute_risk_by_route.png", dpi=150, facecolor=SURFACE)
    print("wrote docs/commute_risk_by_route.png")


NOTES = """## Method notes

- Setting: weekdays only, request 05:00-10:59, drop-off in the 6 destination zones; July 2026
  trains, August 2026 tests (strict split on request time). Ride tables every 10 minutes
  (05:00 ... 10:50 and 10:59); requests outside the grid use its nearest edge.
- Lookup: empirical quantiles per (pu_zone, do_zone, req_hour) on July weekday trips in the
  setting, round-1 back-off (>= 10 rows, else route, else pickup zone, else all). LightGBM:
  quantile models on all July weekday trips in the setting, defaults, 4 threads. TabPFN-3.5:
  round-3 retrieval context per (route, hour): all own-route rows within +-2 request hours
  (truncated to 05-10), then pu_borough -> do_zone rows up to 10,000; GPU only.
- Alarm: the pickup-scenario Monte Carlo (core.Scenario) with the meeting as the deadline;
  latest safe wake by bisection over the 4 hours before the meeting.
- Gut rule median ride: lookup q50 for the route at the request hour of meeting - 30 min.
- Reality: prep ~ N(15, 5) clipped at 0 and a uniform u, shared by all strategies; the ride
  is the u-quantile of real trips on the same August date and route requested within
  +-15 min of the strategy's request time (wake + prep), else +-30 min. A scenario is kept
  only if every strategy found trips for every draw (so all strategies are scored on the
  same scenarios); skipped scenarios are counted above.
- Minutes early = meeting - arrival when on time, 0 when late.

## Judgment calls

- Destinations exclude airports (JFK and LGA are among the busiest morning drop-offs but are
  not workplaces) and unknown zones 264/265.
- "Residential" origins have no TLC label; they are every zone except the destinations,
  airports and 264/265. The dense rule (most July weekday 06:00-09:59 trips) picks four short
  Queens hops into Jamaica (130), one Upper East Side -> Midtown and one Canarsie -> East New
  York; it was applied as written.
- Sparse routes: 10-29 July trips, one borough at a time (boroughs ordered by their best
  candidate), ranked by August trips so the same-date reality pool is as full as possible.
- Weekdays are Monday-Friday; Friday 2026-07-03 (Independence Day observed) stays in training.
- The ride table grid covers request 05:00-10:59; the latest safe wake is searched over the
  4 hours before the meeting.
- The gut-rule curve used for the gap is every B from 0 to 30 minutes; the reported gut
  strategies are B in {0, 5, 10, 15, 20, 30}.
- Risk spread: weights per route = kept scenarios, recomputed in every bootstrap replicate
  (a route with no scenario on the drawn days has weight 0). MAD10 (mean distance from 10%)
  answers "closer to 10%"; the fixed-buffer comparator is the gut B whose overall late rate
  is closest to 10%.

## Fairness

- 541 of 630 sparse scenarios were skipped: a sparse route rarely has a real trip on the
  same date within +-30 min. The 89 kept sparse scenarios are the busier days and times of
  those routes, so the sparse results are a small, selected sample with wide intervals.
- Lookup and LightGBM use all July rows in the setting; TabPFN uses a 10k retrieval context
  per route and hour. Nothing was tuned.
- The reality pool is the same date and route, so it carries that day's traffic, which no
  model saw; it is the same for every strategy.
- One month of weekdays (21 days) and 12 routes; four of the six dense routes end in Jamaica.
"""


if __name__ == "__main__":
    {"commute-routes": choose_routes, "tables": tables, "run": run,
     "report": report}[sys.argv[1]]()
