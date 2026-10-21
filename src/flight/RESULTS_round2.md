# Flight workstream, round 2: regime features (holdout 2026-08)

Round 1 (`RESULTS.md`) found every model's P90 too low in 2026-08 (coverage 0.82-0.86 against
0.90) because the month had a heavier late tail than most training months. Round 2 adds
information known the night before about the current "regime" and re-runs the same comparison.

Setup, unchanged from round 1: ArrDelay (minutes) of flown flights into JFK/LGA scheduled to
arrive 03:00-08:59 (`crs_arr_hhmm` 300-859). Train = flights dated 2025-08 ... 2026-07 (28,672
flown rows), test = flights dated 2026-08 (2,487 flown rows). Every model and every variant is
fit on the same 28,672 rows and scored on the same 2,487 rows. Defaults, no tuning. TabPFN
uses all 28,672 rows as context.

## Direct answer

**Does P90 coverage on 2026-08 move toward 0.90 with the new features?** Yes, for TabPFN and
LightGBM, and the change is larger than its CI. But the pinball loss does not improve, and the
gain does not come from the features telling bad days apart from calm ones.

| model | P90 cov without | P90 cov with | paired change [95% day-bootstrap CI] |
|---|---|---|---|
| TabPFN-3.5 | 0.854 | 0.881 | +0.027 [+0.013, +0.046] |
| LightGBM quantile | 0.820 | 0.842 | +0.021 [+0.006, +0.037] |
| lookups (cannot use the features) | 0.858 / 0.861 | same | 0 |

- Both changes exclude zero. TabPFN moves in every bin with at least 50 training flights
  (+0.018 to +0.042). LightGBM moves in all of them too (+0.016 to +0.034), but its 200-799 bin
  CI touches zero. Neither model reaches 0.90: TabPFN's CI [0.854, 0.905] includes it,
  LightGBM's [0.811, 0.868] does not.
- **The new feature that moves coverage is the previous-day pair, not dest.** `dest only`
  (round-1 features + dest) leaves TabPFN's P90 coverage unchanged (+0.001) and LightGBM's
  slightly lower (-0.010).
- **Mean pinball loss does not improve.** TabPFN with minus without is -0.010
  [-0.214, +0.263]. pin10 and pin50 get a little better and pin90 a little worse
  (+0.251 [-0.453, +1.348]). For LightGBM the change is -0.094 [-0.262, +0.070]. The
  only significant pinball gain in round 2 comes from **dest alone, for TabPFN**: -0.030
  [-0.048, -0.013].
- **The previous-day features do not track the August late tail.** The observed P90 of
  ArrDelay is about the same after calm, middle and bad A-2 days (52.2 / 47.5 / 50.4 min).
  In 2026-08 the Spearman correlation of ArrDelay with the features is +0.036 (origin) and
  +0.019 (dest); in training it was +0.060 / +0.088. Daily mean arrival delay at JFK barely
  persists two days (autocorrelation 0.19 at lag 2, 0.45 at lag 1; LGA 0.08 / 0.20).
  What the features do carry is the level: in 2026-08, 31 of the 63 JFK/LGA arrival-days
  in the test set follow a "bad" A-2 day (top training quartile, >= 17.6 min) and only 7 follow
  a "calm" one (bottom quartile). By construction each group holds 25% in training. The
  monthly mean of the daily JFK/LGA arrival delay was 24.7 min in 2026-08 and 29.1 in
  2026-07, against 0.8-20.9 in the other training months. So the models raise their P90 on
  most August days ("this looks like last July"). That fixes part of the coverage gap without
  ranking the days, which matches the flat pinball loss.
- **TabPFN-specific artefact.** A day-level feature takes one value per airport and day, so it
  can act as a near-unique key for a training day. On 2026-08-23 (JFK, A-2 mean 15.53 min),
  TabPFN `with` predicts a median P90 of 486 min (range
  190-737) for the 56 JFK flights that morning. Actual P90 that morning: 101 min. The nearest training value is JFK 2026-02-24 (15.56 min), a
  storm-recovery morning with 6 flown flights and a median delay of 443 min. 64 TabPFN
  `with` rows have P90 > 150 min, on 3 days (LightGBM `with`: 14 rows on 9 days; no `without`
  or `dest only` row of either model). With those 3 days left out (post hoc), TabPFN's
  coverage gain is +0.018 [+0.008, +0.027]: smaller, still above zero.
- For the pickup decision, the lower tail matters more than P90. You are late when the flight
  lands *earlier* than you reach the curb. TabPFN `with` lowers P10 coverage from 0.099 to
  0.089 (-0.010 [-0.019, -0.002]), so its P10 is now slightly too low, which is the safe side.
  But a wrongly inflated P50 (as on 2026-08-23, P50 up to 82 min against an actual P50 near 0)
  is the unsafe direction for a snooze rule.

**Runtime choice that follows:** `flight_arrival_samples` now uses TabPFN with the `dest only`
feature set by default. The previous-day variant is available with `SNOOZE_FLIGHT_VARIANT=with`.
Three reasons: it has no pinball gain, it shows the day-key artefact, and its features cannot be
filled for a real upcoming morning from BTS (published weeks later). This choice was made
after seeing the 2026-08 results (see Judgment calls).

## Regime features (`regime.py`)

For a flight scheduled to arrive on NYC-local date A (the date of `crs_arr_ts`):

- `origin_prev_dep_delay`: mean `DepDelay` over **all** BTS departures from the flight's origin
  (any destination, any carrier) with `FlightDate` = A - 2.
- `dest_prev_arr_delay`: mean `ArrDelay` over **all** BTS arrivals at the flight's dest (JFK or
  LGA) whose scheduled arrival date (NYC local) is A - 2.

Aggregated with DuckDB straight from `data/raw/bts/ontime_YYYY-MM.csv` (memory_limit 2 GB,
4 threads) into `data/flight2/airport_day.parquet` (143,925 origin-days, 855 JFK/LGA-days).
Why A - 2, cancelled/diverted handling and missing days: see Judgment calls 1-5.

Missing values: 8 training rows lack `origin_prev_dep_delay` and 67 lack `dest_prev_arr_delay`.
The 67 are JFK/LGA days with no flown arrival at all, e.g. storm days. No test row is missing
either feature.

## Results

Cells are the point value with a 95% day-bootstrap CI: 2,000 resamples of the 32 scheduled
arrival dates (2026-08-01 ... 2026-09-01). Each statistic is the sum over the drawn days divided
by the row count over the drawn days. Columns as in round 1: P10/P90 cov = share of flights
with `arr_delay <=` the predicted quantile (targets 0.10 / 0.90); pin10/50/90 = mean pinball
loss in minutes; pin mean = their average. Bins = flown training flights per origin x carrier
(round-1 bins; 0 and 1-49 merged). Schedule and lookups are identical in every variant: they
cannot use the new features.

Variants:
- `without`: round-1 features (origin, carrier, crs_arr_min_of_day, dow, month, distance).
  TabPFN's `without` predictions reproduce round 1 exactly (max abs difference 0.0 over all
  2,487 x 203 quantiles).
- `with`: `without` + dest + origin_prev_dep_delay + dest_prev_arr_delay (what the task asked).
- `dest only`: `without` + dest. A supplementary decomposition, added after the first two
  were scored, to separate dest from the previous-day features.

#### Variant `without`

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| schedule (delay=0) | <50 (incl. unseen) | 47 | 0.766 [0.629, 0.877] | 0.766 [0.629, 0.877] | 13.915 [10.974, 16.710] | 12.979 [8.407, 19.822] | 12.043 [3.726, 25.125] | 12.979 [8.407, 19.822] |
| lookup origin | <50 (incl. unseen) | 47 | 0.149 [0.058, 0.244] | 0.915 [0.822, 0.980] | 3.495 [2.146, 5.249] | 10.574 [5.811, 18.000] | 10.293 [2.903, 22.457] | 8.121 [3.764, 15.160] |
| lookup origin x carrier | <50 (incl. unseen) | 47 | 0.128 [0.043, 0.222] | 0.894 [0.787, 0.977] | 3.204 [1.944, 4.900] | 10.181 [5.448, 17.711] | 10.206 [2.616, 22.651] | 7.863 [3.430, 15.063] |
| LightGBM quantile | <50 (incl. unseen) | 47 | 0.191 [0.078, 0.317] | 0.915 [0.822, 0.980] | 3.594 [2.294, 5.264] | 10.702 [5.965, 18.146] | 10.090 [3.057, 21.880] | 8.129 [3.955, 14.972] |
| TabPFN-3.5 | <50 (incl. unseen) | 47 | 0.191 [0.089, 0.298] | 0.915 [0.822, 0.980] | 3.557 [2.392, 5.084] | 11.261 [6.425, 18.757] | 11.235 [3.755, 23.406] | 8.684 [4.317, 15.709] |
| schedule (delay=0) | 50-199 | 476 | 0.662 [0.600, 0.721] | 0.662 [0.600, 0.721] | 13.572 [12.340, 14.853] | 20.661 [16.342, 25.597] | 27.749 [19.618, 37.056] | 20.661 [16.342, 25.597] |
| lookup origin | 50-199 | 476 | 0.074 [0.045, 0.103] | 0.836 [0.792, 0.876] | 4.843 [3.894, 5.964] | 19.102 [14.519, 24.318] | 21.005 [14.173, 29.082] | 14.984 [10.872, 19.721] |
| lookup origin x carrier | 50-199 | 476 | 0.080 [0.051, 0.109] | 0.847 [0.800, 0.891] | 4.919 [3.955, 6.051] | 19.117 [14.491, 24.311] | 20.515 [13.703, 28.635] | 14.850 [10.759, 19.598] |
| LightGBM quantile | 50-199 | 476 | 0.149 [0.112, 0.190] | 0.807 [0.752, 0.858] | 4.893 [3.938, 6.015] | 19.018 [14.456, 24.150] | 21.227 [14.208, 29.378] | 15.046 [10.867, 19.847] |
| TabPFN-3.5 | 50-199 | 476 | 0.101 [0.070, 0.130] | 0.834 [0.785, 0.883] | 4.848 [3.894, 5.971] | 19.023 [14.405, 24.167] | 20.330 [13.512, 28.469] | 14.734 [10.609, 19.544] |
| schedule (delay=0) | 200-799 | 1580 | 0.692 [0.640, 0.738] | 0.692 [0.640, 0.738] | 11.144 [10.228, 12.003] | 14.048 [11.929, 16.204] | 16.952 [12.668, 21.225] | 14.048 [11.929, 16.204] |
| lookup origin | 200-799 | 1580 | 0.082 [0.064, 0.100] | 0.860 [0.828, 0.889] | 3.829 [3.326, 4.335] | 13.224 [10.821, 15.679] | 13.803 [10.345, 17.329] | 10.285 [8.155, 12.443] |
| lookup origin x carrier | 200-799 | 1580 | 0.084 [0.066, 0.100] | 0.863 [0.831, 0.891] | 3.803 [3.293, 4.317] | 13.138 [10.715, 15.592] | 13.580 [10.211, 17.010] | 10.174 [8.062, 12.304] |
| LightGBM quantile | 200-799 | 1580 | 0.120 [0.095, 0.144] | 0.826 [0.789, 0.858] | 3.790 [3.281, 4.282] | 13.018 [10.627, 15.410] | 13.758 [10.233, 17.360] | 10.189 [8.062, 12.321] |
| TabPFN-3.5 | 200-799 | 1580 | 0.101 [0.078, 0.125] | 0.857 [0.826, 0.886] | 3.781 [3.279, 4.275] | 12.954 [10.577, 15.336] | 13.357 [9.974, 16.858] | 10.031 [7.958, 12.116] |
| schedule (delay=0) | 800+ | 384 | 0.672 [0.616, 0.728] | 0.672 [0.616, 0.728] | 10.160 [9.137, 11.370] | 11.017 [9.902, 12.206] | 11.874 [8.995, 14.787] | 11.017 [9.902, 12.206] |
| lookup origin | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.824, 0.909] | 3.284 [2.900, 3.688] | 10.327 [8.687, 11.998] | 8.065 [6.274, 10.003] | 7.225 [5.990, 8.528] |
| lookup origin x carrier | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.826, 0.907] | 3.323 [2.936, 3.728] | 10.306 [8.680, 11.946] | 7.938 [6.232, 9.818] | 7.189 [5.978, 8.451] |
| LightGBM quantile | 800+ | 384 | 0.089 [0.059, 0.122] | 0.802 [0.761, 0.844] | 3.121 [2.744, 3.513] | 9.981 [8.407, 11.638] | 7.916 [6.096, 9.889] | 7.006 [5.779, 8.327] |
| TabPFN-3.5 | 800+ | 384 | 0.076 [0.046, 0.105] | 0.859 [0.817, 0.899] | 3.066 [2.690, 3.460] | 9.919 [8.372, 11.559] | 7.433 [5.749, 9.329] | 6.806 [5.683, 8.071] |
| schedule (delay=0) | all | 2487 | 0.684 [0.636, 0.729] | 0.684 [0.636, 0.729] | 11.509 [10.634, 12.327] | 14.826 [12.907, 16.782] | 18.142 [14.228, 22.236] | 14.826 [12.907, 16.782] |
| lookup origin | all | 2487 | 0.078 [0.061, 0.095] | 0.858 [0.827, 0.886] | 3.933 [3.469, 4.429] | 13.852 [11.661, 16.174] | 14.229 [11.192, 17.435] | 10.671 [8.812, 12.699] |
| lookup origin x carrier | all | 2487 | 0.080 [0.065, 0.094] | 0.861 [0.830, 0.889] | 3.931 [3.466, 4.428] | 13.789 [11.592, 16.110] | 13.973 [10.977, 17.126] | 10.564 [8.720, 12.568] |
| LightGBM quantile | all | 2487 | 0.122 [0.099, 0.146] | 0.820 [0.786, 0.851] | 3.894 [3.452, 4.369] | 13.654 [11.500, 15.919] | 14.216 [11.153, 17.469] | 10.588 [8.736, 12.582] |
| TabPFN-3.5 | all | 2487 | 0.099 [0.079, 0.118] | 0.854 [0.823, 0.882] | 3.870 [3.422, 4.361] | 13.615 [11.456, 15.888] | 13.737 [10.769, 16.922] | 10.407 [8.606, 12.361] |

#### Variant `with`

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| schedule (delay=0) | <50 (incl. unseen) | 47 | 0.766 [0.629, 0.877] | 0.766 [0.629, 0.877] | 13.915 [10.974, 16.710] | 12.979 [8.407, 19.822] | 12.043 [3.726, 25.125] | 12.979 [8.407, 19.822] |
| lookup origin | <50 (incl. unseen) | 47 | 0.149 [0.058, 0.244] | 0.915 [0.822, 0.980] | 3.495 [2.146, 5.249] | 10.574 [5.811, 18.000] | 10.293 [2.903, 22.457] | 8.121 [3.764, 15.160] |
| lookup origin x carrier | <50 (incl. unseen) | 47 | 0.128 [0.043, 0.222] | 0.894 [0.787, 0.977] | 3.204 [1.944, 4.900] | 10.181 [5.448, 17.711] | 10.206 [2.616, 22.651] | 7.863 [3.430, 15.063] |
| LightGBM quantile | <50 (incl. unseen) | 47 | 0.213 [0.100, 0.333] | 0.915 [0.830, 0.980] | 3.651 [2.327, 5.316] | 10.990 [6.240, 18.212] | 9.498 [2.977, 20.573] | 8.047 [4.045, 14.604] |
| TabPFN-3.5 | <50 (incl. unseen) | 47 | 0.191 [0.078, 0.317] | 0.915 [0.822, 0.980] | 3.424 [2.243, 5.027] | 11.177 [6.403, 18.433] | 11.786 [4.521, 23.003] | 8.795 [4.516, 15.492] |
| schedule (delay=0) | 50-199 | 476 | 0.662 [0.600, 0.721] | 0.662 [0.600, 0.721] | 13.572 [12.340, 14.853] | 20.661 [16.342, 25.597] | 27.749 [19.618, 37.056] | 20.661 [16.342, 25.597] |
| lookup origin | 50-199 | 476 | 0.074 [0.045, 0.103] | 0.836 [0.792, 0.876] | 4.843 [3.894, 5.964] | 19.102 [14.519, 24.318] | 21.005 [14.173, 29.082] | 14.984 [10.872, 19.721] |
| lookup origin x carrier | 50-199 | 476 | 0.080 [0.051, 0.109] | 0.847 [0.800, 0.891] | 4.919 [3.955, 6.051] | 19.117 [14.491, 24.311] | 20.515 [13.703, 28.635] | 14.850 [10.759, 19.598] |
| LightGBM quantile | 50-199 | 476 | 0.139 [0.103, 0.174] | 0.840 [0.791, 0.887] | 4.902 [3.933, 6.029] | 19.136 [14.559, 24.273] | 20.649 [13.899, 28.520] | 14.896 [10.829, 19.573] |
| TabPFN-3.5 | 50-199 | 476 | 0.097 [0.066, 0.126] | 0.876 [0.833, 0.915] | 4.814 [3.885, 5.906] | 18.543 [14.211, 23.370] | 19.851 [13.462, 27.526] | 14.403 [10.522, 18.933] |
| schedule (delay=0) | 200-799 | 1580 | 0.692 [0.640, 0.738] | 0.692 [0.640, 0.738] | 11.144 [10.228, 12.003] | 14.048 [11.929, 16.204] | 16.952 [12.668, 21.225] | 14.048 [11.929, 16.204] |
| lookup origin | 200-799 | 1580 | 0.082 [0.064, 0.100] | 0.860 [0.828, 0.889] | 3.829 [3.326, 4.335] | 13.224 [10.821, 15.679] | 13.803 [10.345, 17.329] | 10.285 [8.155, 12.443] |
| lookup origin x carrier | 200-799 | 1580 | 0.084 [0.066, 0.100] | 0.863 [0.831, 0.891] | 3.803 [3.293, 4.317] | 13.138 [10.715, 15.592] | 13.580 [10.211, 17.010] | 10.174 [8.062, 12.304] |
| LightGBM quantile | 200-799 | 1580 | 0.119 [0.094, 0.143] | 0.842 [0.810, 0.870] | 3.772 [3.266, 4.263] | 12.992 [10.635, 15.355] | 13.637 [10.229, 17.122] | 10.134 [8.048, 12.224] |
| TabPFN-3.5 | 200-799 | 1580 | 0.089 [0.070, 0.108] | 0.882 [0.855, 0.907] | 3.727 [3.238, 4.220] | 12.759 [10.415, 15.122] | 13.785 [10.248, 17.377] | 10.090 [8.021, 12.180] |
| schedule (delay=0) | 800+ | 384 | 0.672 [0.616, 0.728] | 0.672 [0.616, 0.728] | 10.160 [9.137, 11.370] | 11.017 [9.902, 12.206] | 11.874 [8.995, 14.787] | 11.017 [9.902, 12.206] |
| lookup origin | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.824, 0.909] | 3.284 [2.900, 3.688] | 10.327 [8.687, 11.998] | 8.065 [6.274, 10.003] | 7.225 [5.990, 8.528] |
| lookup origin x carrier | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.826, 0.907] | 3.323 [2.936, 3.728] | 10.306 [8.680, 11.946] | 7.938 [6.232, 9.818] | 7.189 [5.978, 8.451] |
| LightGBM quantile | 800+ | 384 | 0.091 [0.057, 0.129] | 0.831 [0.787, 0.873] | 3.086 [2.706, 3.473] | 9.961 [8.470, 11.499] | 7.410 [5.743, 9.196] | 6.819 [5.696, 7.993] |
| TabPFN-3.5 | 800+ | 384 | 0.068 [0.040, 0.095] | 0.878 [0.842, 0.911] | 3.056 [2.680, 3.448] | 9.808 [8.291, 11.357] | 7.823 [5.936, 9.975] | 6.896 [5.737, 8.160] |
| schedule (delay=0) | all | 2487 | 0.684 [0.636, 0.729] | 0.684 [0.636, 0.729] | 11.509 [10.634, 12.327] | 14.826 [12.907, 16.782] | 18.142 [14.228, 22.236] | 14.826 [12.907, 16.782] |
| lookup origin | all | 2487 | 0.078 [0.061, 0.095] | 0.858 [0.827, 0.886] | 3.933 [3.469, 4.429] | 13.852 [11.661, 16.174] | 14.229 [11.192, 17.435] | 10.671 [8.812, 12.699] |
| lookup origin x carrier | all | 2487 | 0.080 [0.065, 0.094] | 0.861 [0.830, 0.889] | 3.931 [3.466, 4.428] | 13.789 [11.592, 16.110] | 13.973 [10.977, 17.126] | 10.564 [8.720, 12.568] |
| LightGBM quantile | all | 2487 | 0.120 [0.097, 0.143] | 0.842 [0.811, 0.868] | 3.880 [3.435, 4.365] | 13.662 [11.503, 15.895] | 13.939 [10.897, 17.037] | 10.494 [8.655, 12.400] |
| TabPFN-3.5 | all | 2487 | 0.089 [0.073, 0.104] | 0.881 [0.854, 0.905] | 3.826 [3.383, 4.299] | 13.380 [11.291, 15.586] | 13.988 [10.820, 17.311] | 10.398 [8.555, 12.400] |

#### Variant `dest only`

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| schedule (delay=0) | <50 (incl. unseen) | 47 | 0.766 [0.629, 0.877] | 0.766 [0.629, 0.877] | 13.915 [10.974, 16.710] | 12.979 [8.407, 19.822] | 12.043 [3.726, 25.125] | 12.979 [8.407, 19.822] |
| lookup origin | <50 (incl. unseen) | 47 | 0.149 [0.058, 0.244] | 0.915 [0.822, 0.980] | 3.495 [2.146, 5.249] | 10.574 [5.811, 18.000] | 10.293 [2.903, 22.457] | 8.121 [3.764, 15.160] |
| lookup origin x carrier | <50 (incl. unseen) | 47 | 0.128 [0.043, 0.222] | 0.894 [0.787, 0.977] | 3.204 [1.944, 4.900] | 10.181 [5.448, 17.711] | 10.206 [2.616, 22.651] | 7.863 [3.430, 15.063] |
| LightGBM quantile | <50 (incl. unseen) | 47 | 0.213 [0.100, 0.333] | 0.915 [0.822, 0.980] | 3.720 [2.346, 5.419] | 10.781 [5.983, 18.353] | 10.156 [2.708, 22.330] | 8.219 [3.869, 15.318] |
| TabPFN-3.5 | <50 (incl. unseen) | 47 | 0.213 [0.095, 0.340] | 0.915 [0.822, 0.980] | 3.557 [2.380, 5.091] | 11.156 [6.311, 18.576] | 11.265 [3.782, 23.443] | 8.660 [4.290, 15.669] |
| schedule (delay=0) | 50-199 | 476 | 0.662 [0.600, 0.721] | 0.662 [0.600, 0.721] | 13.572 [12.340, 14.853] | 20.661 [16.342, 25.597] | 27.749 [19.618, 37.056] | 20.661 [16.342, 25.597] |
| lookup origin | 50-199 | 476 | 0.074 [0.045, 0.103] | 0.836 [0.792, 0.876] | 4.843 [3.894, 5.964] | 19.102 [14.519, 24.318] | 21.005 [14.173, 29.082] | 14.984 [10.872, 19.721] |
| lookup origin x carrier | 50-199 | 476 | 0.080 [0.051, 0.109] | 0.847 [0.800, 0.891] | 4.919 [3.955, 6.051] | 19.117 [14.491, 24.311] | 20.515 [13.703, 28.635] | 14.850 [10.759, 19.598] |
| LightGBM quantile | 50-199 | 476 | 0.141 [0.105, 0.175] | 0.792 [0.736, 0.844] | 4.912 [3.945, 6.039] | 19.255 [14.666, 24.416] | 21.434 [14.290, 29.817] | 15.200 [10.953, 20.028] |
| TabPFN-3.5 | 50-199 | 476 | 0.107 [0.075, 0.140] | 0.834 [0.785, 0.883] | 4.852 [3.904, 5.971] | 18.986 [14.385, 24.122] | 20.294 [13.512, 28.377] | 14.711 [10.605, 19.497] |
| schedule (delay=0) | 200-799 | 1580 | 0.692 [0.640, 0.738] | 0.692 [0.640, 0.738] | 11.144 [10.228, 12.003] | 14.048 [11.929, 16.204] | 16.952 [12.668, 21.225] | 14.048 [11.929, 16.204] |
| lookup origin | 200-799 | 1580 | 0.082 [0.064, 0.100] | 0.860 [0.828, 0.889] | 3.829 [3.326, 4.335] | 13.224 [10.821, 15.679] | 13.803 [10.345, 17.329] | 10.285 [8.155, 12.443] |
| lookup origin x carrier | 200-799 | 1580 | 0.084 [0.066, 0.100] | 0.863 [0.831, 0.891] | 3.803 [3.293, 4.317] | 13.138 [10.715, 15.592] | 13.580 [10.211, 17.010] | 10.174 [8.062, 12.304] |
| LightGBM quantile | 200-799 | 1580 | 0.120 [0.095, 0.143] | 0.816 [0.777, 0.851] | 3.805 [3.303, 4.290] | 13.008 [10.621, 15.405] | 14.022 [10.457, 17.686] | 10.278 [8.121, 12.447] |
| TabPFN-3.5 | 200-799 | 1580 | 0.102 [0.078, 0.125] | 0.859 [0.828, 0.887] | 3.762 [3.268, 4.253] | 12.915 [10.545, 15.287] | 13.332 [9.951, 16.834] | 10.003 [7.930, 12.080] |
| schedule (delay=0) | 800+ | 384 | 0.672 [0.616, 0.728] | 0.672 [0.616, 0.728] | 10.160 [9.137, 11.370] | 11.017 [9.902, 12.206] | 11.874 [8.995, 14.787] | 11.017 [9.902, 12.206] |
| lookup origin | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.824, 0.909] | 3.284 [2.900, 3.688] | 10.327 [8.687, 11.998] | 8.065 [6.274, 10.003] | 7.225 [5.990, 8.528] |
| lookup origin x carrier | 800+ | 384 | 0.057 [0.031, 0.086] | 0.867 [0.826, 0.907] | 3.323 [2.936, 3.728] | 10.306 [8.680, 11.946] | 7.938 [6.232, 9.818] | 7.189 [5.978, 8.451] |
| LightGBM quantile | 800+ | 384 | 0.078 [0.045, 0.113] | 0.799 [0.755, 0.843] | 3.100 [2.717, 3.498] | 10.031 [8.462, 11.647] | 7.997 [6.150, 10.061] | 7.043 [5.817, 8.373] |
| TabPFN-3.5 | 800+ | 384 | 0.076 [0.046, 0.106] | 0.859 [0.817, 0.899] | 3.054 [2.685, 3.444] | 9.867 [8.345, 11.490] | 7.346 [5.730, 9.187] | 6.755 [5.662, 7.995] |
| schedule (delay=0) | all | 2487 | 0.684 [0.636, 0.729] | 0.684 [0.636, 0.729] | 11.509 [10.634, 12.327] | 14.826 [12.907, 16.782] | 18.142 [14.228, 22.236] | 14.826 [12.907, 16.782] |
| lookup origin | all | 2487 | 0.078 [0.061, 0.095] | 0.858 [0.827, 0.886] | 3.933 [3.469, 4.429] | 13.852 [11.661, 16.174] | 14.229 [11.192, 17.435] | 10.671 [8.812, 12.699] |
| lookup origin x carrier | all | 2487 | 0.080 [0.065, 0.094] | 0.861 [0.830, 0.889] | 3.931 [3.466, 4.428] | 13.789 [11.592, 16.110] | 13.973 [10.977, 17.126] | 10.564 [8.720, 12.568] |
| LightGBM quantile | all | 2487 | 0.119 [0.096, 0.141] | 0.811 [0.775, 0.844] | 3.907 [3.466, 4.384] | 13.702 [11.534, 15.971] | 14.437 [11.280, 17.746] | 10.682 [8.801, 12.701] |
| TabPFN-3.5 | all | 2487 | 0.101 [0.081, 0.120] | 0.855 [0.825, 0.883] | 3.858 [3.416, 4.343] | 13.573 [11.428, 15.837] | 13.701 [10.754, 16.893] | 10.377 [8.585, 12.323] |

#### Paired difference: `with` minus `without` (same test rows)

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| TabPFN-3.5 | <50 (incl. unseen) | 47 | +0.000 [-0.060, +0.061] | +0.000 [+0.000, +0.000] | -0.133 [-0.308, +0.018] | -0.084 [-0.416, +0.288] | +0.551 [-0.972, +2.707] | +0.111 [-0.465, +0.958] |
| TabPFN-3.5 | 50-199 | 476 | -0.004 [-0.026, +0.012] | +0.042 [+0.019, +0.071] | -0.034 [-0.090, +0.017] | -0.480 [-1.226, +0.001] | -0.479 [-1.357, +0.351] | -0.331 [-0.830, +0.104] |
| TabPFN-3.5 | 200-799 | 1580 | -0.013 [-0.021, -0.005] | +0.025 [+0.010, +0.045] | -0.054 [-0.093, -0.016] | -0.196 [-0.496, -0.014] | +0.428 [-0.420, +1.833] | +0.060 [-0.193, +0.445] |
| TabPFN-3.5 | 800+ | 384 | -0.008 [-0.019, +0.003] | +0.018 [+0.003, +0.037] | -0.009 [-0.063, +0.040] | -0.111 [-0.413, +0.130] | +0.391 [-0.561, +1.812] | +0.090 [-0.230, +0.503] |
| TabPFN-3.5 | all | 2487 | -0.010 [-0.019, -0.002] | +0.027 [+0.013, +0.046] | -0.045 [-0.082, -0.010] | -0.235 [-0.603, -0.015] | +0.251 [-0.453, +1.348] | -0.010 [-0.214, +0.263] |
| LightGBM quantile | <50 (incl. unseen) | 47 | +0.021 [+0.000, +0.067] | +0.000 [-0.060, +0.067] | +0.058 [-0.051, +0.181] | +0.288 [-0.125, +0.725] | -0.591 [-1.559, +0.144] | -0.082 [-0.459, +0.202] |
| LightGBM quantile | 50-199 | 476 | -0.011 [-0.025, +0.004] | +0.034 [+0.006, +0.063] | +0.008 [-0.037, +0.054] | +0.118 [-0.066, +0.298] | -0.578 [-1.849, +0.586] | -0.150 [-0.627, +0.281] |
| LightGBM quantile | 200-799 | 1580 | -0.001 [-0.009, +0.007] | +0.016 [-0.001, +0.034] | -0.017 [-0.041, +0.006] | -0.026 [-0.124, +0.075] | -0.122 [-0.473, +0.230] | -0.055 [-0.193, +0.081] |
| LightGBM quantile | 800+ | 384 | +0.003 [-0.010, +0.013] | +0.029 [+0.003, +0.058] | -0.035 [-0.070, +0.005] | -0.021 [-0.265, +0.201] | -0.506 [-1.086, -0.074] | -0.187 [-0.438, +0.011] |
| LightGBM quantile | all | 2487 | -0.002 [-0.007, +0.004] | +0.021 [+0.006, +0.037] | -0.014 [-0.034, +0.006] | +0.008 [-0.103, +0.116] | -0.277 [-0.699, +0.129] | -0.094 [-0.262, +0.070] |

#### Paired difference: `dest only` minus `without` (same test rows)

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| TabPFN-3.5 | <50 (incl. unseen) | 47 | +0.021 [+0.000, +0.070] | +0.000 [+0.000, +0.000] | +0.001 [-0.024, +0.032] | -0.105 [-0.166, -0.035] | +0.031 [-0.051, +0.122] | -0.024 [-0.063, +0.019] |
| TabPFN-3.5 | 50-199 | 476 | +0.006 [-0.002, +0.016] | +0.000 [+0.000, +0.000] | +0.004 [-0.014, +0.022] | -0.037 [-0.070, -0.003] | -0.036 [-0.141, +0.052] | -0.023 [-0.065, +0.014] |
| TabPFN-3.5 | 200-799 | 1580 | +0.001 [-0.003, +0.004] | +0.002 [-0.001, +0.005] | -0.019 [-0.026, -0.012] | -0.039 [-0.058, -0.021] | -0.024 [-0.059, +0.007] | -0.027 [-0.044, -0.012] |
| TabPFN-3.5 | 800+ | 384 | +0.000 [-0.008, +0.008] | +0.000 [+0.000, +0.000] | -0.012 [-0.025, +0.002] | -0.052 [-0.107, +0.000] | -0.087 [-0.201, +0.011] | -0.051 [-0.102, -0.003] |
| TabPFN-3.5 | all | 2487 | +0.002 [-0.001, +0.005] | +0.001 [-0.001, +0.003] | -0.013 [-0.020, -0.006] | -0.042 [-0.063, -0.022] | -0.035 [-0.072, -0.004] | -0.030 [-0.048, -0.013] |
| LightGBM quantile | <50 (incl. unseen) | 47 | +0.021 [+0.000, +0.067] | +0.000 [+0.000, +0.000] | +0.127 [+0.012, +0.252] | +0.078 [-0.106, +0.258] | +0.067 [-0.457, +0.785] | +0.091 [-0.108, +0.364] |
| LightGBM quantile | 50-199 | 476 | -0.008 [-0.020, +0.002] | -0.015 [-0.029, +0.000] | +0.019 [-0.024, +0.057] | +0.237 [+0.130, +0.334] | +0.207 [-0.168, +0.612] | +0.154 [+0.013, +0.310] |
| LightGBM quantile | 200-799 | 1580 | -0.001 [-0.006, +0.005] | -0.010 [-0.021, +0.001] | +0.016 [+0.002, +0.030] | -0.011 [-0.038, +0.017] | +0.264 [+0.092, +0.455] | +0.090 [+0.030, +0.156] |
| LightGBM quantile | 800+ | 384 | -0.010 [-0.023, +0.002] | -0.003 [-0.018, +0.012] | -0.021 [-0.049, +0.005] | +0.050 [-0.032, +0.133] | +0.081 [-0.034, +0.217] | +0.037 [-0.003, +0.085] |
| LightGBM quantile | all | 2487 | -0.003 [-0.007, +0.001] | -0.010 [-0.019, -0.002] | +0.013 [-0.001, +0.026] | +0.048 [+0.016, +0.077] | +0.221 [+0.096, +0.353] | +0.094 [+0.047, +0.140] |

#### Paired difference: `with` minus `dest only` (same test rows)

| model | bin | n_test | P10 cov | P90 cov | pin10 | pin50 | pin90 | pin mean |
|---|---|---|---|---|---|---|---|---|
| TabPFN-3.5 | <50 (incl. unseen) | 47 | -0.021 [-0.068, +0.000] | +0.000 [+0.000, +0.000] | -0.134 [-0.310, +0.008] | +0.020 [-0.307, +0.404] | +0.521 [-1.005, +2.683] | +0.136 [-0.451, +0.989] |
| TabPFN-3.5 | 50-199 | 476 | -0.011 [-0.033, +0.006] | +0.042 [+0.019, +0.071] | -0.038 [-0.088, +0.009] | -0.443 [-1.174, +0.016] | -0.443 [-1.253, +0.367] | -0.308 [-0.774, +0.100] |
| TabPFN-3.5 | 200-799 | 1580 | -0.013 [-0.022, -0.005] | +0.023 [+0.009, +0.041] | -0.035 [-0.072, +0.000] | -0.156 [-0.454, +0.026] | +0.453 [-0.392, +1.842] | +0.087 [-0.161, +0.471] |
| TabPFN-3.5 | 800+ | 384 | -0.008 [-0.017, +0.000] | +0.018 [+0.003, +0.037] | +0.003 [-0.051, +0.054] | -0.059 [-0.356, +0.201] | +0.478 [-0.477, +1.946] | +0.141 [-0.192, +0.576] |
| TabPFN-3.5 | all | 2487 | -0.012 [-0.021, -0.004] | +0.026 [+0.012, +0.044] | -0.032 [-0.067, +0.000] | -0.193 [-0.562, +0.026] | +0.286 [-0.427, +1.402] | +0.021 [-0.185, +0.300] |
| LightGBM quantile | <50 (incl. unseen) | 47 | +0.000 [+0.000, +0.000] | +0.000 [-0.060, +0.067] | -0.069 [-0.234, +0.089] | +0.209 [-0.267, +0.679] | -0.658 [-2.248, +0.417] | -0.172 [-0.775, +0.238] |
| LightGBM quantile | 50-199 | 476 | -0.002 [-0.015, +0.012] | +0.048 [+0.021, +0.079] | -0.010 [-0.078, +0.054] | -0.119 [-0.293, +0.049] | -0.784 [-2.138, +0.447] | -0.305 [-0.807, +0.143] |
| LightGBM quantile | 200-799 | 1580 | -0.001 [-0.007, +0.007] | +0.027 [+0.010, +0.043] | -0.033 [-0.055, -0.013] | -0.016 [-0.120, +0.090] | -0.386 [-0.796, -0.003] | -0.145 [-0.309, +0.003] |
| LightGBM quantile | 800+ | 384 | +0.013 [+0.003, +0.024] | +0.031 [+0.005, +0.061] | -0.014 [-0.050, +0.023] | -0.070 [-0.286, +0.117] | -0.587 [-1.210, -0.113] | -0.224 [-0.480, -0.020] |
| LightGBM quantile | all | 2487 | +0.001 [-0.003, +0.006] | +0.031 [+0.015, +0.047] | -0.026 [-0.045, -0.009] | -0.040 [-0.151, +0.064] | -0.498 [-0.968, -0.061] | -0.188 [-0.377, -0.011] |

#### P90 coverage by regime of day A-2 at the destination

Groups by `dest_prev_arr_delay` (mean ArrDelay at JFK/LGA two days before arrival); thresholds = quartiles of that daily mean over training-period JFK/LGA days: calm < -2.1 min, bad >= 17.6 min.

| A-2 regime at dest | n_test | test days | observed P90 ArrDelay | model | variant | P90 cov | median predicted P90 |
|---|---|---|---|---|---|---|---|
| calm (bottom quartile) | 272 | 4 | 52.200 | lookup origin x carrier (no regime) | without | 0.875 [0.753, 0.939] | 16.600 |
| calm (bottom quartile) | 272 | 4 | 52.200 | lookup origin x carrier (no regime) | dest only | 0.875 [0.753, 0.939] | 16.600 |
| calm (bottom quartile) | 272 | 4 | 52.200 | LightGBM quantile | without | 0.831 [0.704, 0.900] | 11.000 |
| calm (bottom quartile) | 272 | 4 | 52.200 | LightGBM quantile | with | 0.790 [0.654, 0.879] | 6.500 |
| calm (bottom quartile) | 272 | 4 | 52.200 | LightGBM quantile | dest only | 0.824 [0.679, 0.939] | 10.300 |
| calm (bottom quartile) | 272 | 4 | 52.200 | TabPFN-3.5 | without | 0.864 [0.741, 0.939] | 15.300 |
| calm (bottom quartile) | 272 | 4 | 52.200 | TabPFN-3.5 | with | 0.857 [0.716, 0.939] | 11.600 |
| calm (bottom quartile) | 272 | 4 | 52.200 | TabPFN-3.5 | dest only | 0.868 [0.741, 0.939] | 15.100 |
| middle | 906 | 16 | 47.500 | lookup origin x carrier (no regime) | without | 0.855 [0.790, 0.915] | 18.100 |
| middle | 906 | 16 | 47.500 | lookup origin x carrier (no regime) | dest only | 0.855 [0.790, 0.915] | 18.100 |
| middle | 906 | 16 | 47.500 | LightGBM quantile | without | 0.816 [0.749, 0.876] | 11.900 |
| middle | 906 | 16 | 47.500 | LightGBM quantile | with | 0.826 [0.764, 0.882] | 14.200 |
| middle | 906 | 16 | 47.500 | LightGBM quantile | dest only | 0.808 [0.737, 0.875] | 11.000 |
| middle | 906 | 16 | 47.500 | TabPFN-3.5 | without | 0.858 [0.789, 0.917] | 14.700 |
| middle | 906 | 16 | 47.500 | TabPFN-3.5 | with | 0.882 [0.825, 0.923] | 15.900 |
| middle | 906 | 16 | 47.500 | TabPFN-3.5 | dest only | 0.860 [0.791, 0.917] | 14.700 |
| bad (top quartile) | 1309 | 18 | 50.400 | lookup origin x carrier (no regime) | without | 0.862 [0.831, 0.893] | 19.000 |
| bad (top quartile) | 1309 | 18 | 50.400 | lookup origin x carrier (no regime) | dest only | 0.862 [0.831, 0.893] | 19.000 |
| bad (top quartile) | 1309 | 18 | 50.400 | LightGBM quantile | without | 0.821 [0.785, 0.859] | 14.900 |
| bad (top quartile) | 1309 | 18 | 50.400 | LightGBM quantile | with | 0.863 [0.835, 0.889] | 25.000 |
| bad (top quartile) | 1309 | 18 | 50.400 | LightGBM quantile | dest only | 0.810 [0.772, 0.848] | 12.600 |
| bad (top quartile) | 1309 | 18 | 50.400 | TabPFN-3.5 | without | 0.850 [0.820, 0.880] | 15.100 |
| bad (top quartile) | 1309 | 18 | 50.400 | TabPFN-3.5 | with | 0.885 [0.856, 0.915] | 24.200 |
| bad (top quartile) | 1309 | 18 | 50.400 | TabPFN-3.5 | dest only | 0.850 [0.820, 0.880] | 15.200 |

| feature | Spearman with ArrDelay, train | Spearman with ArrDelay, test 2026-08 | missing train / test |
|---|---|---|---|
| origin_prev_dep_delay | +0.060 | +0.036 | 8 / 0 |
| dest_prev_arr_delay | +0.088 | +0.019 | 67 / 0 |

Daily mean ArrDelay at JFK/LGA, 2025-07 ... 2026-08:

| airport | autocorr lag 1 day | lag 2 days | lag 3 days |
|---|---|---|---|
| JFK | 0.454 | 0.186 | 0.134 |
| LGA | 0.197 | 0.079 | 0.053 |

#### Post-hoc check: predicted P90 above 150 min

| model | variant | rows with q90 > cut | days |
|---|---|---|---|
| LightGBM quantile | dest only | 0 |  |
| LightGBM quantile | with | 14 | 2026-08-08, 2026-08-09, 2026-08-10, 2026-08-12, 2026-08-13, 2026-08-22, 2026-08-24, 2026-08-26, 2026-08-29 |
| LightGBM quantile | without | 0 |  |
| TabPFN-3.5 | dest only | 0 |  |
| TabPFN-3.5 | with | 64 | 2026-08-13, 2026-08-22, 2026-08-23 |
| TabPFN-3.5 | without | 0 |  |

P90 coverage change, all test days vs without those days:

| model | pair | all days | without flagged days |
|---|---|---|---|
| TabPFN-3.5 | with - without | +0.027 [+0.013, +0.046] | +0.018 [+0.008, +0.027] |
| TabPFN-3.5 | dest only - without | +0.001 [-0.001, +0.003] | +0.001 [-0.001, +0.003] |
| TabPFN-3.5 | with - dest only | +0.026 [+0.012, +0.044] | +0.017 [+0.008, +0.026] |
| LightGBM quantile | with - without | +0.021 [+0.006, +0.037] | +0.021 [+0.004, +0.038] |
| LightGBM quantile | dest only - without | -0.010 [-0.019, -0.002] | -0.009 [-0.019, -0.000] |
| LightGBM quantile | with - dest only | +0.031 [+0.015, +0.047] | +0.030 [+0.013, +0.048] |

## p_cancel (optional item)

Scored on all 2,630 scheduled 03:00-08:59 arrivals dated 2026-08 (observed rate 0.0502),
`python -m src.flight.cancel 2` -> `data/flight2/cancel_calibration2.json`.

| model | mean predicted | Brier | log loss |
|---|---|---|---|
| constant training rate | 0.0333 | 0.04796 | 0.2029 |
| p_cancel round 1 (origin x carrier) | 0.0335 | 0.04765 | 0.1990 |
| p_cancel with dest (origin x carrier x dest, shrunk to origin x carrier) | 0.0340 | 0.04765 | 0.1993 |
| logistic regression on logit(p_cancel with dest) + both regime features (report only) | 0.0369 | 0.04759 | 0.1993 |

Cancellation rate by the A-2 regime at the destination (same training-quartile thresholds,
-2.1 / 17.6 min):

| | calm | middle | bad |
|---|---|---|---|
| train (2025-08 ... 2026-07) | 0.0235 (n 8,221) | 0.0323 (14,162) | 0.0463 (7,260) |
| test 2026-08 | 0.0882 (306) | 0.0310 (936) | 0.0548 (1,388) |

The dest level changes nothing measurable. In training the regime features have some
relation to cancellations (calm 2.3% vs bad 4.6%), but on 2026-08 the relation does not hold
(the calm group has the highest rate, from few days). The logistic recalibration moves the
mean prediction up a little (0.0369) and gains 0.00006 Brier and no log loss. Not
worth it. p_cancel's design is unchanged except for the optional dest level, which
`p_cancel(..., dest=...)` now applies (dest=None returns the round-1 value).

## Runtime interface (`model.py`)

- `flight_arrival_samples(origin, carrier, crs_arr_ts, n=2000, dest=None, seed=0)` and
  `p_cancel(origin, carrier, crs_arr_ts, dest=None)`, per SPEC.
- dest known: features include dest, and distance is the exact route distance (most
  common training distance for origin -> dest; fallback to round 1's guess for unseen routes).
- dest None: the flight is predicted for both JFK and LGA. The samples are a mixture: each
  sample's airport is drawn with the training share of the origin x carrier cell's flights to
  JFK vs LGA (else the origin's share, else the overall share).
- `precompute(queries)` takes the GPU lock itself, only around fit + predict, and frees GPU
  memory before releasing it. Do not wrap the process in `flock` on the same file: it would
  wait on itself.
- Cache: `data/flight2/grid_cache_<variant>.parquet`, keyed by (origin, carrier, crs_arr_ts,
  dest).

Smoke test (`python -m src.flight.model`, default `dest only`): one realistic nightly batch with
all 85 flown test flights arriving 2026-08-26, plus one flight without dest and one unknown
origin/carrier. The batch is 87 queries, 88 predicted rows; TabPFN is called once.

- Took 240 s wall clock, about 190 s of it inside the GPU lock. Peak RSS 2.94 GB.
- The runtime grids match the evaluation grids for the same rows to within 0.017 min at
  P10/P50/P90. The largest difference anywhere on the 203-point grid is 3.1 min, at the
  extreme tail points (0.001/0.999). TabPFN output depends slightly on the batch it is
  predicted in.

| query (origin, carrier, crs_arr_ts, dest) | sample P10 / P50 / P90 | p_cancel |
|---|---|---|
| SAN, B6, 2026-08-26 05:03, JFK | -37.5 / -21.5 / +8.1 | 0.0389 |
| CLT, YX, 2026-08-26 08:56, LGA | -24.7 / -14.7 / +4.9 | 0.0229 |
| ATL, DL, 2026-08-26 07:00, None (JFK/LGA mixture) | -24.4 / -12.2 / +10.4 | 0.0286 |
| XXX, ZZ, 2026-08-26 06:00, LGA (unknown) | -25.9 / -12.8 / +14.0 | 0.0333 |

**Small batches are slow on the shared GPU.** Two earlier smoke attempts, each with a
5-row batch, were still predicting after 10 min and 5.5 min inside the lock (fit itself took
1.2 s). I killed both to release the lock. The cause was not found; it is probably contention
with the other GPU jobs. 500-row chunks took 70-213 s and the 88-row batch about 190 s, so the
nightly precompute should send all flights in one batch, never one at a time.

## How to reproduce

From the worktree root, with
`PY=/home/elias-shieh/projects/competition/hackathon/snooze-oracle/.venv/bin/python`.
The tabpfn stages take the GPU lock per chunk themselves, so do not wrap them in `flock`:

```bash
$PY -m src.flight.regime                                   # airport-day table, ~1.5 min CPU
$PY -m src.flight.evaluate2 tabpfn --variant without       # 5 chunks x ~1.5-3.5 min GPU
$PY -m src.flight.evaluate2 tabpfn --variant with
$PY -m src.flight.evaluate2 tabpfn --variant "dest only"
$PY -m src.flight.evaluate2 baselines                      # lookups + LightGBM, seconds
$PY -m src.flight.evaluate2 report     # writes data/flight2/results_round2.md
$PY -m src.flight.cancel 2             # writes data/flight2/cancel_calibration2.json
$PY -m src.flight.model                # interface smoke test (takes the GPU lock itself)
```

Round-1 commands in `RESULTS.md` still work and still write to `data/flight/`.

**Extra BTS download:** `regime.py` downloads 2025-07
(`https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_7.zip`,
32 MB) with `download()` from `scripts/prep_trips.py` into `data/raw/bts/` and extracts
`ontime_2025-07.csv`. It is needed so that training arrivals on 2025-08-01 and 2025-08-02 have
their A - 2 day. No processed `flights_2025-07.parquet` is written.

**Resources** (TabPFN per chunk of 500 test rows; each chunk refits (~2 s) and predicts inside
the GPU lock, then frees GPU memory):

| variant | GPU time inside the lock, 5 chunks | peak GPU alloc / reserved |
|---|---|---|
| with | 523 s | 2,886 / 4,080 MiB |
| without | 536 s | 2,563 / 4,062 MiB |
| dest only | 526 s | 2,670 / 3,532 MiB |

Total TabPFN GPU time for the evaluation: 1,585 s (26.4 min). The longest single hold was
213 s. Add the smoke test (~190 s) and the two aborted small-batch smoke attempts (~10 min and
~5.5 min inside the lock): about 2,700 s (45 min) of GPU-lock time in round 2. Waits for the lock (other jobs) reached 31 min for one chunk. Peak RSS (VmHWM) of each
TabPFN process: 3.82 GB, of which about 1.3 GB is file-backed (memory-mapped CUDA/torch
libraries); anonymous RSS sampled after each chunk was 0.25-1.59 GB. The HWM rose from
2.97 GB to 3.82 GB at the second chunk, so the per-chunk refit costs about 0.85 GB of peak.
This is above the 3 GB budget; see Fairness/notes. Other stages: regime build 1.72 GB
(DuckDB), baselines 0.39 GB, report 0.41 GB, cancel 0.35 GB.

## Judgment calls

Choices that SPEC.md and the task do not dictate:

1. **Lag = A - 2, anchored on the scheduled arrival date A (NYC local).** The interface only
   receives `crs_arr_ts`, so the date must come from it. The departure date would need the
   schedule, and for same-day departures its previous day is A - 1. Predictions are
   precomputed the evening of A - 1, and the previous calendar day of that precompute is
   A - 2, the last day whose flights have all departed and landed by then. A - 1 is still
   running at precompute time. Worse, BTS `FlightDate` A - 1 contains the red-eyes that land
   on the morning of A, i.e. the target flights themselves.
2. **Day definitions.** Origin: BTS `FlightDate` (origin-local departure date), the natural
   "day at the origin". Destination: scheduled arrival date in NYC local, computed with the
   same red-eye rule as `prep_flights.py` (`FlightDate` + 1 if `CRSArrTime < CRSDepTime` or
   = 2400): the natural "day at JFK/LGA".
3. **Cancelled and diverted flights.** Means use flights with a non-null delay. Cancelled
   flights have no DepDelay/ArrDelay and drop out of both means. Diverted flights departed, so
   they stay in the origin mean, and they never arrived, so they drop out of the destination
   mean. Cancellation counts are not used as a feature (the task named two features). A day
   with mass cancellations can therefore show a modest mean delay.
4. **No clipping.** Raw `DepDelay`/`ArrDelay`, early flights count as negative (not
   `DepDelayMinutes`). Small-count days are not shrunk.
5. **No qualifying flights on A - 2 -> NaN**, left for the models (TabPFN and LightGBM both take
   missing values natively), rather than imputed. The flight count is stored next to the
   mean in `airport_day.parquet` but not used as a feature.
6. **dest as a categorical feature** (integer code flagged categorical for TabPFN, pandas
   category for LightGBM), like origin and carrier.
7. **Distance in the evaluation is the row's own BTS distance in every variant** (as in round 1),
   so "exact distance with dest" only changes the runtime path, not the evaluation.
8. **Bootstrap CIs for every table cell and for the paired differences**: day-block bootstrap
   over scheduled-arrival dates, 2,000 resamples, seed 0. The same resample weights are used
   for every cell.
9. **Regime groups for item 4**: quartiles of the daily JFK/LGA mean arrival delay over
   training-period days (2025-08-01 ... 2026-07-31), fixed from training data:
   calm < -2.1 min, bad >= 17.6 min. Test rows are grouped by their own `dest_prev_arr_delay`.
10. **`dest only` variant** added after seeing the with/without results, to split the effect of
    dest from the effect of the previous-day features. It costs one more TabPFN run (526 s GPU).
11. **Runtime default = `dest only`**: chosen after seeing the 2026-08 results. Reasons: no
    pinball gain from the previous-day features; the day-key artefact; and their inputs are
    unavailable from BTS for a real upcoming morning. `SNOOZE_FLIGHT_VARIANT=with` switches.
12. **The post-hoc artefact check** (P90 > 150 min, coverage without the 3 flagged days) is a
    diagnostic chosen after looking at the predictions, not a pre-registered metric.
13. **TabPFN per-chunk refit**: one fresh `TabPFNRegressor` per 500-row chunk, fit + predict
    inside the GPU lock, then `del` + `torch.cuda.empty_cache()`. This keeps each lock hold to
    1-4 minutes and frees GPU memory between chunks. The `without` run reproduces round 1
    bit for bit, so this does not change predictions.
14. **p_cancel with dest**: one more shrinkage level (origin x carrier x dest -> origin x
    carrier, K = 100). The regime logistic regression (C = 1, sklearn default) is report-only
    and not wired into `p_cancel`. Missing regime values are filled with training medians there.
15. **dest None at runtime -> JFK/LGA mixture** weighted by the training share of the cell (see
    Runtime interface), not a single "unknown dest" prediction. The model never saw a missing
    dest.
16. **`precompute` takes the GPU lock itself** (round 1 asked the caller to wrap the process in
    `flock`). That way it holds the lock only around fit + predict.
17. **Smoke test = one realistic nightly batch** (all flown test flights of 2026-08-26 plus two
    synthetic queries), checked against the evaluation grid. It replaced round 1's 3-query smoke
    test after 5-row batches proved pathologically slow on the shared GPU.
18. **Round-1 code kept runnable**: `evaluate.py` and its outputs in `data/flight/` are untouched.
    `data.Encoder` gained optional `features`/`cats` arguments, with the round-1 defaults.

## Fairness

- **Lookups cannot use the new features.** Schedule-only and both lookups are the same in all
  variants. Any with-vs-without gain is available only to TabPFN and LightGBM by design. The
  lookups' P90 coverage (0.858 / 0.861) now sits between LightGBM-with (0.842) and TabPFN-with
  (0.881).
- **The new features carry a month-level signal that matches this particular test month.**
  2026-08 looked like 2026-07 (high daily mean delays), and 2026-07 was also the heavy-tail
  training month. In a test month whose regime does not resemble a training month, the
  coverage gain may not repeat. One test month cannot separate "regime awareness" from "August
  looked like July".
- **TabPFN is more exposed to day-key matching than LightGBM.** Day-level features act as
  near-unique keys, and TabPFN matches test rows to similar context rows, so it copied one
  storm day's delays (2026-08-23). LightGBM's splits are coarser. This cuts both ways: it
  inflates TabPFN's P90 coverage on those days and worsens its pin90.
- **Defaults are not equally suited to the new features.** Neither model was tuned. LightGBM's
  defaults (min_child_samples 20) limit how finely it can split on a day-level feature, and
  TabPFN has no such limit.
- **The `dest only` variant and the runtime choice were added after seeing results.** The
  required with/without comparison was fixed beforehand. The decomposition and the default
  choice are post hoc.
- **CIs are per cell and per difference, not adjusted for multiple comparisons** (3 variants x
  2 models x 5 bins x 6 metrics). Treat single small effects with a CI just off zero as weak.
- **Test rows identical across variants and models**: asserted in code (`arr_delay` equal row
  by row for every paired difference).
- **Resource note, not a fairness issue:** TabPFN processes peaked at 3.82 GB RSS (about
  1.3 GB of it file-backed CUDA libraries), above the 3 GB budget. They never came close
  to OOM: about 10 GB was available.

## Proposed SPEC.md changes (for main, not applied here)

- State that regime/previous-day features use the day **two** days before the scheduled
  arrival date (the previous calendar day of a night-before precompute), and that their live
  values need a data source other than BTS (BTS is published monthly with a lag).
- `p_cancel` and `flight_arrival_samples` with `dest=None`: document the JFK/LGA mixture.
- If precompute moves to after midnight on A, A - 1 becomes partly available (JFK daily mean
  delay autocorrelation 0.45 at lag 1 vs 0.19 at lag 2). That would need an "up to time t"
  feature definition, not a calendar-day one.
- The GPU lock: say whether processes take `/tmp/snooze-oracle-gpu.lock` themselves per batch
  (round 2) or are wrapped in `flock` (round 1). Mixing the two in one process deadlocks.
