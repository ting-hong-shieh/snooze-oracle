"""End-to-end backtest on 2026-08: wake-time strategies vs what really happened.

Usage (from the repo root):
    python -m src.alarm.backtest routes   # writes data/ride3/backtest_routes.csv
    python -m src.alarm.tables flight
    python -m src.alarm.tables ride --routes data/ride3/backtest_routes.csv
    python -m src.alarm.backtest run      # scenarios, strategies, reality draws (CPU)
    python -m src.alarm.backtest report   # day-bootstrap tables + docs/frontier.png

Scenarios: every flight into JFK/LGA scheduled 04:00-07:59 (crs_arr_hhmm 400-759) in the
2026-08 test month (FlightDate in August, as in the flight test split) that is also
scheduled to land in August (red-eyes landing on 09-01 are counted and dropped; red-eyes
landing on 08-01 departed in July, a training month, and are not in the test set) x the 8
pickup zones
chosen for its airport (data/ride3/backtest_routes.csv). Cancelled and diverted flights are
counted and skipped.

Reality, per scenario, REALITY_DRAWS draws shared by every strategy (common random numbers):
    prep  ~ N(15, 5), clipped at 0
    ride  = the u-quantile (u ~ U(0, 1), linear interpolation) of real August trips on the
            same route whose request minute of day is within +-60 min of the strategy's
            request time (wake + prep, rounded to the minute), any day in August
    delay = the flight's actual ArrDelay
    late  <=> wake + prep + ride > crs_arr + delay + exit_buffer
    wait  = max(0, crs_arr + delay + exit_buffer - (wake + prep + ride))  (minutes the
            driver waits at the airport; 0 when late)
A scenario is skipped (and counted) if any strategy's request time has no trips in its pool.

Strategies (one wake time per scenario, chosen the night before):
    gut_B       crs_arr + exit_buffer - prep_min - lookup median ride - B, B in GUT_B;
                the median is the July lookup q50 for the route at the request hour of
                crs_arr + exit_buffer - 30 min (clamped to 03:00-08:59)
    schedule    delay = 0: crs_arr + exit_buffer - prep_min - lookup ride P90 (same hour)
    lookup / lgbm / tabpfn   latest wake with p_on_time >= 0.90 (src.alarm.core)
    tabpfn_snooze   alarm rings 30 min before the tabpfn wake time; the user snoozes
                whenever allowed (largest s <= 9 min with p_on_time(now + s) >= 0.90) and
                gets up at the first refusal
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from .core import Scenario, THRESHOLD
from .tables import LCOLS, LEVELS, OUT, RIDE_GRID

ROOT = OUT.parents[1]
ROUTES = ROOT / "data" / "ride3" / "backtest_routes.csv"
GUT_B = (0, 15, 30, 45, 60, 90)
# Reference curve only: gut rules every 5 min, so the curve a strategy is compared with is
# not a few chords of a convex curve (chords lie above it and flatter any strategy).
GUT_CURVE = tuple(range(0, 91, 5))
PREP_MIN, PREP_SD, EXIT_BUFFER = 15.0, 5.0, 20.0
REALITY_DRAWS = 5
POOL_RADIUS = 60
SEED = 20261021
MODELS = ("lookup", "lgbm", "tabpfn")


def choose_routes() -> pd.DataFrame:
    """Per airport: the 4 pickup zones with the most July trips in the setting (dense) and
    the 4 with 10-29 July trips that have the most August trips (sparse, ties by zone id).
    Airport zones (1, 132, 138) are not pickup zones."""
    from src.ride.retrieval import TEST_FILE, connect, load_train, window_sql, zone_borough

    jul = load_train().groupby(["pu_zone", "do_zone"]).size().rename("jul")
    aug = connect().execute(window_sql(TEST_FILE, "2026-08-01", "2026-09-01")).fetchdf()
    aug = aug.groupby(["pu_zone", "do_zone"]).size().rename("aug")
    d = pd.concat([jul, aug], axis=1).fillna(0).astype(int).reset_index()
    d = d[~d["pu_zone"].isin([1, 132, 138])]
    parts = []
    for do in (132, 138):
        x = d[d["do_zone"] == do]
        parts.append(x.sort_values("jul", ascending=False).head(4).assign(kind="dense"))
        parts.append(x[x["jul"].between(10, 29)].sort_values(["aug", "pu_zone"],
                     ascending=[False, True]).head(4).assign(kind="sparse"))
    r = pd.concat(parts)
    names = pd.read_csv(ROOT / "data" / "raw" / "taxi_zone_lookup.csv").set_index("LocationID")
    r["zone"] = r["pu_zone"].map(names["Zone"])
    r["borough"] = r["pu_zone"].map(zone_borough())
    ROUTES.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(ROUTES, index=False)
    return r


def flights() -> tuple[pd.DataFrame, dict]:
    from src.flight.data import TEST_MONTHS, load_months

    f = load_months(TEST_MONTHS)
    f = f[f["crs_arr_hhmm"].between(400, 759)].reset_index(drop=True)
    sep = pd.to_datetime(f["crs_arr_ts"]).dt.month != 8
    counts = {"arrive_after_august": int(sep.sum())}
    f = f[~sep].reset_index(drop=True)
    counts |= {"scheduled": len(f), "cancelled": int(f["cancelled"].sum()),
              "diverted": int((f["diverted"] & ~f["cancelled"]).sum())}
    f = f[f["flown"]].reset_index(drop=True)
    counts["flown"] = len(f)
    return f, counts


class RealityPool:
    """Real August trips per route, sorted ride totals per request-minute window."""

    def __init__(self, routes: list[tuple[int, int]]):
        from src.ride.retrieval import TEST_FILE, connect

        r = ", ".join(f"({a}, {b})" for a, b in routes)
        df = connect().execute(f"""
            SELECT pu_zone, do_zone, req_min_of_day, total_min FROM read_parquet('{TEST_FILE}')
            WHERE (pu_zone, do_zone) IN ({r})
              AND request_ts >= TIMESTAMP '2026-08-01' AND request_ts < TIMESTAMP '2026-09-01'
        """).fetchdf()
        self.by_route = {k: (g["req_min_of_day"].to_numpy(), g["total_min"].to_numpy())
                         for k, g in df.groupby(["pu_zone", "do_zone"])}
        self.cache: dict = {}

    def window(self, route: tuple[int, int], minute: int) -> np.ndarray:
        key = (route, minute)
        if key not in self.cache:
            m, v = self.by_route.get(route, (np.array([]), np.array([])))
            d = np.abs(m - (minute % 1440))
            d = np.minimum(d, 1440 - d)
            self.cache[key] = np.sort(v[d <= POOL_RADIUS])
        return self.cache[key]

    def draw(self, route, request_min: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = np.full(len(u), np.nan)
        for i, (t, uu) in enumerate(zip(np.rint(request_min).astype(int), u)):
            w = self.window(route, int(t))
            if len(w):
                out[i] = np.quantile(w, uu)
        return out


def lookup_ride_quantile(table: dict, route, date, minute: float, level: float) -> float:
    q = table[(route[0], route[1], date)]
    t = int(np.argmin(np.abs(RIDE_GRID - np.clip(minute, RIDE_GRID[0], RIDE_GRID[-1]))))
    return float(np.interp(level, LEVELS, q[t]))


def snooze(s: Scenario, wake: float) -> tuple[float, int, int]:
    """(wake minute, snoozes granted, refused) for an alarm first ringing 30 min early."""
    now, granted = wake - 30.0, 0
    while True:
        ok = s.p_on_time(now + np.arange(1, 10)) >= THRESHOLD
        if not ok.any():
            return now, granted, 1
        now += int(np.flatnonzero(ok)[-1]) + 1
        granted += 1


def run() -> None:
    from .core import _ride_table

    t_start = time.time()
    routes = pd.read_csv(ROUTES)
    f, counts = flights()
    pool = RealityPool(list(zip(routes.pu_zone, routes.do_zone)))
    lk = _ride_table("lookup")
    rows, skipped, infeasible = [], {"dense": 0, "sparse": 0}, {m: 0 for m in MODELS}
    for fi, fl in f.iterrows():
        do = {"JFK": 132, "LGA": 138}[fl.dest]
        ts = pd.Timestamp(fl.crs_arr_ts)
        date, crs = ts.date(), ts.hour * 60 + ts.minute
        flight = {"origin": fl.origin, "carrier": fl.carrier, "dest": fl.dest, "crs_arr_ts": ts}
        for _, rt in routes[routes.do_zone == do].iterrows():
            route = (int(rt.pu_zone), do)
            sid = f"{fi}_{rt.pu_zone}"
            guess = crs + EXIT_BUFFER - 30
            med = lookup_ride_quantile(lk, route, date, guess, 0.5)
            p90 = lookup_ride_quantile(lk, route, date, guess, 0.9)
            wake = {f"gut_{b}": crs + EXIT_BUFFER - PREP_MIN - med - b for b in GUT_CURVE}
            wake["schedule"] = crs + EXIT_BUFFER - PREP_MIN - p90
            extra = {}
            for m in MODELS:
                s = Scenario(route[0], flight, m, PREP_MIN, PREP_SD, EXIT_BUFFER,
                             seed=SEED + fi)
                w, feasible = s.latest_safe_wake()
                infeasible[m] += not feasible
                wake[m] = w
                if m == "tabpfn":
                    ws, g, r = snooze(s, w)
                    wake["tabpfn_snooze"] = ws
                    extra = {"snooze_granted": g, "snooze_refused": r}
            rng = np.random.default_rng([SEED, fi, int(rt.pu_zone)])
            prep = np.maximum(rng.normal(PREP_MIN, PREP_SD, REALITY_DRAWS), 0.0)
            u = rng.uniform(0, 1, REALITY_DRAWS)
            ready = crs + fl.arr_delay + EXIT_BUFFER
            res = {}
            for k, w in wake.items():
                ride = pool.draw(route, w + prep, u)
                res[k] = (w, ride)
            if any(np.isnan(r).any() for _, r in res.values()):
                skipped[rt.kind] += 1
                continue
            for k, (w, ride) in res.items():
                arrive = w + prep + ride
                for j in range(REALITY_DRAWS):
                    rows.append((sid, str(date), rt.kind, int(rt.pu_zone), fl.dest, k, w,
                                 bool(arrive[j] > ready),
                                 float(max(0.0, ready - arrive[j])), j,
                                 extra.get("snooze_granted", 0) if k == "tabpfn_snooze" else 0))
        if fi % 200 == 0:
            print(f"{fi}/{len(f)} flights, {time.time() - t_start:.0f}s", flush=True)
    out = pd.DataFrame(rows, columns=["scenario", "date", "kind", "pu_zone", "dest", "strategy",
                                      "wake_min", "late", "wait_min", "draw", "snoozes"])
    out.to_parquet(OUT / "backtest_rows.parquet", index=False)
    meta = {**counts, "scenarios_skipped": skipped, "infeasible": infeasible,
            "scenarios_kept": int(out["scenario"].nunique()), "seconds": round(time.time() - t_start)}
    pd.Series(meta).to_json(OUT / "backtest_meta.json")
    print(meta)


# ------------------------------------------------------------------ report

N_BOOT = 2000
STRATEGIES = [f"gut_{b}" for b in GUT_B] + ["schedule", "lookup", "lgbm", "tabpfn",
                                             "tabpfn_snooze"]
LABEL = {**{f"gut_{b}": f"gut B={b}" for b in GUT_B}, "schedule": "schedule only",
         "lookup": "lookup p>=0.90", "lgbm": "LightGBM p>=0.90",
         "tabpfn": "TabPFN p>=0.90", "tabpfn_snooze": "TabPFN + snooze"}


class Boot:
    """Day-level bootstrap: resample the August arrival days with replacement; every row
    of a drawn day is kept as many times as the day is drawn."""

    def __init__(self, days: np.ndarray):
        self.days = np.unique(days)
        self.idx = np.searchsorted(self.days, days)
        rng = np.random.default_rng(0)
        draws = rng.integers(0, len(self.days), (N_BOOT, len(self.days)))
        self.counts = np.zeros((N_BOOT, len(self.days)))
        np.add.at(self.counts, (np.arange(N_BOOT)[:, None], draws), 1)

    def mean(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Point estimate followed by N_BOOT replicates."""
        nd = len(self.days)
        S = np.bincount(self.idx[mask], weights=x[mask], minlength=nd)
        N = np.bincount(self.idx[mask], minlength=nd).astype(float)
        return np.r_[S.sum() / N.sum(), (self.counts @ S) / (self.counts @ N)]

    def median(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        v, d = x[mask], self.idx[mask]
        o = np.argsort(v, kind="stable")
        v, d = v[o], d[o]
        out = [np.median(v)]
        for c in self.counts:
            w = np.cumsum(c[d])
            out.append(v[np.searchsorted(w, w[-1] / 2)])
        return np.array(out)


def ci(a: np.ndarray, d: int = 3) -> str:
    lo, hi = np.percentile(a[1:], [2.5, 97.5])
    return f"{a[0]:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def gut_curve_gap(stats: dict, s: str) -> np.ndarray:
    """late(s) minus the gut-rule curve's late rate at s's median wait, per replicate
    (linear interpolation between gut points; NaN outside the curve's wait range)."""
    gw = np.stack([stats[f"gut_{b}"]["wait"] for b in GUT_CURVE])  # (B, 1 + N_BOOT)
    gl = np.stack([stats[f"gut_{b}"]["late"] for b in GUT_CURVE])
    out = np.full(gw.shape[1], np.nan)
    for r in range(gw.shape[1]):
        o = np.argsort(gw[:, r])
        x = stats[s]["wait"][r]
        if gw[o[0], r] <= x <= gw[o[-1], r]:
            out[r] = stats[s]["late"][r] - np.interp(x, gw[o, r], gl[o, r])
    return out


def report() -> None:
    rows = pd.read_parquet(OUT / "backtest_rows.parquet")
    meta = pd.read_json(OUT / "backtest_meta.json", typ="series")
    boot = Boot(pd.to_datetime(rows["date"]).dt.day.to_numpy())
    b30 = rows[rows.strategy == "gut_30"].set_index(["scenario", "draw"])["wake_min"]
    rows["wake_vs_b30"] = rows["wake_min"].to_numpy() - b30.loc[
        list(zip(rows.scenario, rows.draw))].to_numpy()
    late, wait, wk = (rows[c].to_numpy(float) for c in ("late", "wait_min", "wake_vs_b30"))
    md = ["# Backtest on 2026-08: wake-time strategies vs reality", "",
          f"Flights scheduled 04:00-07:59: {meta.scheduled} (plus {meta.arrive_after_august} landing on 09-01, dropped), cancelled {meta.cancelled} and "
          f"diverted {meta.diverted} (skipped), flown {meta.flown}. Scenarios kept: "
          f"{meta.scenarios_kept} (flown flights x 8 pickup zones), skipped for an empty "
          f"reality ride pool: {meta.scenarios_skipped}. Wake ranges with no time reaching "
          f"0.90: {meta.infeasible}. {REALITY_DRAWS} reality draws per scenario.",
          "", "95% intervals: day-level bootstrap (2,000 resamples of the 31 arrival days). "
          "Method, judgment calls and fairness notes: `src/alarm/README.md`. "
          "Chart: `docs/frontier.png`.", ""]
    all_stats = {}
    for split in ("all", "dense", "sparse"):
        base = np.ones(len(rows), bool) if split == "all" else (rows["kind"] == split).to_numpy()
        stats = {}
        lines = ["| strategy | late rate | median wait (min) | mean wake vs gut B=30 (min) |",
                 "|---|---|---|---|"]
        for s in STRATEGIES + [f"gut_{b}" for b in GUT_CURVE if b not in GUT_B]:
            m = base & (rows["strategy"] == s).to_numpy()
            stats[s] = {"late": boot.mean(late, m), "wait": boot.median(wait, m),
                        "wake": boot.mean(wk, m), "n": int(m.sum())}
            if s not in STRATEGIES:
                continue
            lines.append(f"| {LABEL[s]} | {ci(stats[s]['late'])} | {ci(stats[s]['wait'], 1)} | "
                         f"{ci(stats[s]['wake'], 1)} |")
        md += [f"## {split} zones", "", *lines, ""]
        md += ["Gap to the gut-rule curve (late rate minus the curve's late rate at the same "
               "median wait, curve = gut rules every 5 min, linear between them; negative = "
               "below the curve):", ""]
        for s in ("schedule", "lookup", "lgbm", "tabpfn", "tabpfn_snooze"):
            g = gut_curve_gap(stats, s)
            ok = ~np.isnan(g[1:])
            txt = (ci(g) if not np.isnan(g[0]) and ok.mean() > 0.95
                   else f"outside the curve's wait range in {1 - ok.mean():.0%} of replicates")
            md.append(f"- {LABEL[s]}: {txt}")
        mE = base & (rows["strategy"] == "tabpfn").to_numpy()
        mC = base & (rows["strategy"] == "lookup").to_numpy()
        mD = base & (rows["strategy"] == "lgbm").to_numpy()
        md += ["", "Paired differences (same scenarios and draws):", ""]
        for name, mo in (("lookup", mC), ("LightGBM", mD)):
            dl = boot.mean(late, mE) - boot.mean(late, mo)
            dw = boot.median(wait, mE) - boot.median(wait, mo)
            dk = boot.mean(wk, mE) - boot.mean(wk, mo)
            md.append(f"- TabPFN minus {name}: late rate {ci(dl)}, median wait {ci(dw, 1)} min, "
                      f"wake time {ci(dk, 1)} min")
        sn = rows[base & (rows["strategy"] == "tabpfn_snooze").to_numpy() & (rows["draw"] == 0)]
        md += ["", f"Snooze simulation: {sn['snoozes'].mean():.2f} snoozes granted per scenario "
                   f"(min {sn['snoozes'].min()}, max {sn['snoozes'].max()}), 1 refusal per scenario "
                   f"(the alarm keeps ringing).", ""]
        all_stats[split] = stats
    (ROOT / "docs" / "backtest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    frontier(all_stats)


def frontier(all_stats: dict) -> None:
    """Top row: the whole gut-rule curve. Bottom row: zoom on the region around gut B=30
    where the model strategies sit. TabPFN + snooze is drawn with TabPFN when identical."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"schedule": "#eb6834", "lookup": "#1baf7a", "lgbm": "#eda100",
             "tabpfn": "#2a78d6", "tabpfn_snooze": "#6250d6"}
    marker = {"schedule": "s", "lookup": "^", "lgbm": "D", "tabpfn": "o", "tabpfn_snooze": "v"}
    zoom_offset = {"lookup": (10, -16), "lgbm": (-118, 10), "tabpfn": (10, 10),
                   "tabpfn_snooze": (10, 24)}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor=surface,
                             gridspec_kw={"height_ratios": [1, 1.1]})
    for col, split in enumerate(("all", "dense", "sparse")):
        st = all_stats[split]
        same = (np.allclose(st["tabpfn"]["late"], st["tabpfn_snooze"]["late"])
                and np.allclose(st["tabpfn"]["wait"], st["tabpfn_snooze"]["wait"]))
        shown = ["schedule", "lookup", "lgbm", "tabpfn"] + ([] if same else ["tabpfn_snooze"])
        for row, ax in enumerate(axes[:, col]):
            ax.set_facecolor(surface)
            ax.plot([st[f"gut_{b}"]["wait"][0] for b in GUT_CURVE],
                    [100 * st[f"gut_{b}"]["late"][0] for b in GUT_CURVE], color=ink2, lw=2,
                    zorder=2, label="gut rules, B = 0 ... 90 min (every 5)")
            gx = [st[f"gut_{b}"]["wait"][0] for b in GUT_B]
            gy = [100 * st[f"gut_{b}"]["late"][0] for b in GUT_B]
            ax.plot(gx, gy, ls="none", marker="o", ms=8, mfc=surface, mec=ink2, mew=2, zorder=2)
            for b, x, y in zip(GUT_B, gx, gy):
                if row == 0 or (14 <= x <= 30):
                    ax.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, 6),
                                fontsize=9, color=ink2)
            for s in shown:
                w, l = st[s]["wait"], 100 * st[s]["late"]
                xerr = [[w[0] - np.percentile(w[1:], 2.5)], [np.percentile(w[1:], 97.5) - w[0]]]
                yerr = [[l[0] - np.percentile(l[1:], 2.5)], [np.percentile(l[1:], 97.5) - l[0]]]
                lab = LABEL[s] + (" (same with snooze)" if s == "tabpfn" and same else "")
                ax.errorbar(w[0], l[0], xerr=xerr, yerr=yerr, fmt=marker[s], ms=9,
                            color=color[s], mec=surface, mew=1.5, ecolor=color[s],
                            elinewidth=1.5, capsize=0, zorder=3, label=lab)
                if row == 1 and s in zoom_offset:
                    ax.annotate(lab, (w[0], l[0]), textcoords="offset points",
                                xytext=zoom_offset[s], fontsize=9, color=ink)
                elif row == 0 and s == "schedule":
                    ax.annotate(lab, (w[0], l[0]), textcoords="offset points", xytext=(10, -4),
                                fontsize=9, color=ink)
            if row == 1:
                ax.set_xlim(14, 30)
                ax.set_ylim(4, 16)
                ax.set_title("zoom: around gut B=30", color=ink2, fontsize=10, loc="left")
            else:
                ax.set_title(f"{split} pickup zones", color=ink, fontsize=12, loc="left")
            ax.set_xlabel("median minutes waiting at the airport", color=ink2)
            ax.grid(color=grid, lw=1)
            ax.tick_params(colors=ink2)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0, col].set_ylabel("late rate (%)" if col == 0 else "", color=ink2)
        axes[1, col].set_ylabel("late rate (%)" if col == 0 else "", color=ink2)
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Picking up an early-morning flight at JFK/LGA, August 2026: late rate vs "
                 "minutes waiting (95% day-bootstrap intervals)", color=ink, fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "frontier.png", dpi=150, facecolor=surface)
    print("wrote docs/frontier.png")

if __name__ == "__main__":
    {"routes": choose_routes, "run": run, "report": report}[sys.argv[1]]()
