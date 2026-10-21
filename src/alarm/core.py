"""Pickup-scenario alarm: p_on_time, snooze_curve, latest_safe_wake (SPEC "Alarm").

Reads only the precomputed tables in data/alarm/ (see tables.py); no model runs here.

Times are minutes after midnight of the flight's scheduled arrival date A (NYC local).
For one scenario (pickup zone, flight) and one model, Monte Carlo with n samples:
    D ~ flight delay quantile function (flown flights only), u_f ~ U(0.01, 0.99)
    P ~ N(prep_min, prep_sd), clipped at 0
    R ~ ride total for a request at w + P: the (route, date A) ride table interpolated
        linearly in request time and in quantile level at u_r ~ U(0.01, 0.99); request
        times outside 03:00-08:59 use the nearest edge of the table
    on time  <=>  w + P + R <= crs_arr + D + exit_buffer
The same (u_f, P, u_r) draws are reused for every wake time w (common random numbers), so
p_on_time is a smooth, nearly monotone function of w.
latest_safe_wake = the latest whole-minute wake time with p_on_time >= threshold, found by
bisection over WAKE_RANGE (minutes relative to the scheduled arrival); it equalled a full
1-minute scan on all 120 sampled scenarios checked.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from .tables import LCOLS, LEVELS, OUT, RIDE_GRID

N_SAMPLES = 2000
WAKE_RANGE = (-300, 120)  # minutes relative to the scheduled arrival
THRESHOLD = 0.90


@lru_cache(maxsize=None)
def _ride_table(model: str) -> dict:
    df = pd.read_parquet(OUT / f"ride_{model}.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out = {}
    for (pu, do, d), g in df.groupby(["pu_zone", "do_zone", "date"]):
        g = g.sort_values("t_min")
        assert (g["t_min"].to_numpy() == RIDE_GRID).all()
        out[(int(pu), int(do), d)] = g[LCOLS].to_numpy()
    return out


@lru_cache(maxsize=None)
def _flight_table(model: str) -> dict:
    df = pd.read_parquet(OUT / f"flight_{model}.parquet")
    return {(o, c, d, pd.Timestamp(t)): v
            for o, c, d, t, v in zip(df.origin, df.carrier, df.dest, df.crs_arr_ts,
                                     df[LCOLS].to_numpy())}


class Scenario:
    """Monte Carlo state for one (pickup zone, flight, model)."""

    def __init__(self, pu_zone: int, flight: dict, model: str = "tabpfn",
                 prep_min: float = 15.0, prep_sd: float = 5.0, exit_buffer: float = 20.0,
                 n: int = N_SAMPLES, seed: int = 0):
        ts = pd.Timestamp(flight["crs_arr_ts"])
        do_zone = {"JFK": 132, "LGA": 138}[flight["dest"]]
        key = (flight["origin"], flight["carrier"], flight["dest"], ts)
        try:
            fq = _flight_table(model)[key]
            self.ride_q = _ride_table(model)[(int(pu_zone), do_zone, ts.date())]
        except KeyError as e:
            raise KeyError(f"not precomputed: {e}") from None
        rng = np.random.default_rng(seed)
        u_f, u_r = rng.uniform(LEVELS[0], LEVELS[-1], (2, n))
        self.delay = np.interp(u_f, LEVELS, fq)
        self.prep = np.maximum(rng.normal(prep_min, prep_sd, n), 0.0)
        # ride level interpolation weights, fixed per sample
        pos = np.interp(u_r, LEVELS, np.arange(len(LEVELS)))
        self.li = np.minimum(pos.astype(int), len(LEVELS) - 2)
        self.lw = pos - self.li
        self.date = ts.normalize()
        self.crs = ts.hour * 60 + ts.minute
        self.ready = self.crs + self.delay + exit_buffer  # passenger out, minutes

    def ride(self, request_min: np.ndarray) -> np.ndarray:
        t = np.clip(request_min, RIDE_GRID[0], RIDE_GRID[-1])
        tp = np.interp(t, RIDE_GRID, np.arange(len(RIDE_GRID)))
        ti = np.minimum(tp.astype(int), len(RIDE_GRID) - 2)
        tw = tp - ti
        Q = self.ride_q
        li, lw = self.li, self.lw
        lo = Q[ti, li] * (1 - lw) + Q[ti, li + 1] * lw
        hi = Q[ti + 1, li] * (1 - lw) + Q[ti + 1, li + 1] * lw
        return lo * (1 - tw) + hi * tw

    def p_on_time(self, wake_min) -> np.ndarray:
        """p_on_time for each wake time (minutes after midnight of date A)."""
        w = np.atleast_1d(np.asarray(wake_min, float))[:, None]
        req = w + self.prep[None, :]
        arrive = req + self.ride(req)
        return (arrive <= self.ready[None, :]).mean(axis=1)

    def latest_safe_wake(self, threshold: float = THRESHOLD) -> tuple[float, bool]:
        """(wake minute, feasible). Bisection on whole minutes; assumes p_on_time is
        non-increasing in the wake time (checked against a full 1-minute scan on a sample
        of scenarios, see scan_latest_safe_wake)."""
        lo, hi = float(self.crs + WAKE_RANGE[0]), float(self.crs + WAKE_RANGE[1])
        if self.p_on_time(lo)[0] < threshold:
            return lo, False
        if self.p_on_time(hi)[0] >= threshold:
            return hi, True
        while hi - lo > 1:  # invariant: p(lo) >= threshold > p(hi)
            mid = float(np.floor((lo + hi) / 2))
            if self.p_on_time(mid)[0] >= threshold:
                lo = mid
            else:
                hi = mid
        return lo, True

    def scan_latest_safe_wake(self, threshold: float = THRESHOLD) -> tuple[float, bool]:
        """Reference: last 1-minute wake time before p_on_time first drops below threshold."""
        w = np.arange(self.crs + WAKE_RANGE[0], self.crs + WAKE_RANGE[1] + 1, 1.0)
        ok = self.p_on_time(w) >= threshold
        if not ok[0]:
            return w[0], False
        bad = np.flatnonzero(~ok)
        return (w[bad[0] - 1] if len(bad) else w[-1]), True

    def to_datetime(self, minute: float) -> datetime:
        return (self.date + timedelta(minutes=float(minute))).to_pydatetime()

    def minute(self, t: datetime) -> float:
        return (pd.Timestamp(t) - self.date).total_seconds() / 60.0


# ------------------------------------------------------------------ public API (SPEC)

def p_on_time(pu_zone: int, flight: dict, wake_time: datetime, prep_min: float = 15.0,
              prep_sd: float = 5.0, exit_buffer: float = 20.0, model: str = "tabpfn") -> float:
    s = Scenario(pu_zone, flight, model, prep_min, prep_sd, exit_buffer)
    return float(s.p_on_time(s.minute(wake_time))[0])


def snooze_curve(pu_zone: int, flight: dict, start: datetime, end: datetime,
                 step_min: int = 1, **kw) -> pd.DataFrame:
    s = Scenario(pu_zone, flight, **kw)
    m = np.arange(s.minute(start), s.minute(end) + 1e-9, step_min)
    return pd.DataFrame({"wake_time": [s.to_datetime(x) for x in m],
                         "p_on_time": s.p_on_time(m)})


def latest_safe_wake(pu_zone: int, flight: dict, threshold: float = THRESHOLD,
                     **kw) -> datetime:
    s = Scenario(pu_zone, flight, **kw)
    m, feasible = s.latest_safe_wake(threshold)
    if not feasible:
        raise ValueError("no wake time in range reaches the threshold")
    return s.to_datetime(m)
