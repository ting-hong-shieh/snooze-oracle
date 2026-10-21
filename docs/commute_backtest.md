# Commute backtest, August 2026 weekdays

Wake up, get ready (prep ~ N(15, 5) min), request an Uber, arrive before the meeting. Strategies pick one wake time the night before; reality uses real trips on the same date and route. Method, judgment calls and fairness notes are at the end.

## Routes

Trips counted on weekday requests 06:00-09:59 (July = training month).

| kind | route | origin | borough | destination | July trips | August trips |
|---|---|---|---|---|---|---|
| dense | 215->130 | South Jamaica | Queens | Jamaica | 3870 | 3351 |
| dense | 10->130 | Baisley Park | Queens | Jamaica | 2260 | 2018 |
| dense | 216->130 | South Ozone Park | Queens | Jamaica | 2070 | 1882 |
| dense | 205->130 | Saint Albans | Queens | Jamaica | 2021 | 1830 |
| dense | 141->161 | Lenox Hill West | Manhattan | Midtown Center | 1878 | 1506 |
| dense | 39->76 | Canarsie | Brooklyn | East New York | 1869 | 2100 |
| sparse | 159->162 | Melrose South | Bronx | Midtown East | 25 | 41 |
| sparse | 83->130 | Elmhurst/Maspeth | Queens | Jamaica | 27 | 39 |
| sparse | 26->76 | Borough Park | Brooklyn | East New York | 27 | 37 |
| sparse | 152->68 | Manhattanville | Manhattan | East Chelsea | 23 | 33 |
| sparse | 204->162 | Rossville/Woodrow | Staten Island | Midtown East | 12 | 14 |
| sparse | 235->161 | University Heights/Morris Heights | Bronx | Midtown Center | 25 | 35 |

Destinations (most July weekday drop-offs requested 07:00-09:59, airports excluded): 161, 246, 162, 130, 68, 76.

## Scenarios

21 August weekdays x 12 routes x 5 meeting times (08:00-10:00) = 1260 scenarios. Skipped because some strategy's request time had no real trip on that date and route within +-30 min: {'dense': 0, 'sparse': 541}. Kept: {'dense': 630, 'sparse': 89}. Reality draws that needed the +-30 min window: 2.9%. Wake ranges with no time reaching 0.90: {'lookup': 0, 'lgbm': 0, 'tabpfn': 0}. 5 reality draws per scenario.

95% intervals: day-level bootstrap (2,000 resamples of the August weekdays).

## all routes (719 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.494 [0.469, 0.523] | 0.2 [0.0, 0.6] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.253 [0.226, 0.287] | 5.3 [4.7, 5.7] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.102 [0.083, 0.125] | 10.4 [9.7, 10.8] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.037 [0.025, 0.051] | 15.3 [14.6, 15.7] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.013 [0.007, 0.019] | 20.4 [19.8, 20.7] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.000 [0.000, 0.001] | 30.5 [29.9, 30.9] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.082 [0.065, 0.101] | 11.2 [10.7, 11.6] | -1.1 [-1.2, -1.0] |
| LightGBM p>=0.90 | 0.068 [0.054, 0.086] | 11.7 [11.2, 12.1] | -1.7 [-1.9, -1.5] |
| TabPFN p>=0.90 | 0.069 [0.056, 0.085] | 11.7 [11.1, 12.2] | -1.6 [-2.0, -1.3] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: -0.004 [-0.009, 0.003]
- LightGBM p>=0.90: -0.009 [-0.013, -0.004]
- TabPFN p>=0.90: -0.008 [-0.014, -0.003]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate -0.013 [-0.020, -0.005], median minutes early 0.6 [0.1, 0.9], wake time -0.5 [-0.9, -0.2] min
- TabPFN minus LightGBM p>=0.90: late rate 0.001 [-0.002, 0.004], median minutes early 0.0 [-0.3, 0.2], wake time 0.1 [-0.1, 0.2] min

## dense routes (630 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.496 [0.475, 0.520] | 0.1 [0.0, 0.5] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.248 [0.223, 0.278] | 5.2 [4.7, 5.6] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.090 [0.076, 0.106] | 10.3 [9.8, 10.7] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.026 [0.019, 0.034] | 15.2 [14.7, 15.6] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.008 [0.005, 0.010] | 20.3 [19.8, 20.7] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.000 [0.000, 0.000] | 30.5 [30.0, 30.8] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.070 [0.058, 0.083] | 11.2 [10.7, 11.6] | -1.0 [-1.1, -1.0] |
| LightGBM p>=0.90 | 0.063 [0.053, 0.076] | 11.4 [11.0, 11.8] | -1.3 [-1.5, -1.1] |
| TabPFN p>=0.90 | 0.062 [0.053, 0.073] | 11.4 [10.8, 11.8] | -1.3 [-1.6, -1.0] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: -0.004 [-0.009, 0.002]
- LightGBM p>=0.90: -0.006 [-0.011, -0.002]
- TabPFN p>=0.90: -0.008 [-0.013, -0.004]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate -0.008 [-0.014, -0.001], median minutes early 0.2 [-0.2, 0.6], wake time -0.3 [-0.6, 0.0] min
- TabPFN minus LightGBM p>=0.90: late rate -0.001 [-0.005, 0.002], median minutes early -0.0 [-0.3, 0.2], wake time -0.0 [-0.2, 0.2] min

## sparse routes (89 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.481 [0.393, 0.571] | 0.5 [0.0, 2.9] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.288 [0.202, 0.388] | 5.6 [3.2, 8.0] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.187 [0.112, 0.280] | 10.7 [8.3, 13.0] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.115 [0.060, 0.179] | 15.6 [13.4, 17.8] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.047 [0.017, 0.089] | 20.4 [18.1, 22.4] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.002 [0.000, 0.008] | 30.5 [27.3, 33.0] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.169 [0.095, 0.255] | 11.5 [9.3, 14.3] | -1.7 [-2.3, -1.1] |
| LightGBM p>=0.90 | 0.101 [0.041, 0.174] | 15.1 [13.4, 17.4] | -4.6 [-5.1, -4.0] |
| TabPFN p>=0.90 | 0.119 [0.061, 0.185] | 14.7 [12.5, 16.8] | -3.9 [-4.8, -3.1] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: 0.002 [-0.030, 0.027]
- LightGBM p>=0.90: -0.019 [-0.047, 0.018]
- TabPFN p>=0.90: -0.006 [-0.037, 0.022]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate -0.049 [-0.088, -0.017], median minutes early 3.2 [1.6, 4.4], wake time -2.2 [-3.1, -1.2] min
- TabPFN minus LightGBM p>=0.90: late rate 0.018 [-0.003, 0.037], median minutes early -0.4 [-1.6, 0.7], wake time 0.6 [0.1, 1.2] min

## Risk by route

Late rate per route, 95% day-bootstrap intervals.

| route | kind | scenarios | gut B=0 | gut B=5 | gut B=10 | gut B=15 | gut B=20 | gut B=30 | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 215->130 | dense | 105 | 52% [48, 56] | 24% [19, 29] | 8% [6, 10] | 2% [1, 4] | 1% [0, 1] | 0% [0, 0] | 8% [6, 10] | 7% [5, 10] | 6% [5, 8] |
| 10->130 | dense | 105 | 47% [42, 52] | 22% [18, 27] | 6% [3, 9] | 1% [0, 2] | 0% [0, 0] | 0% [0, 0] | 6% [3, 8] | 5% [3, 7] | 6% [3, 8] |
| 216->130 | dense | 105 | 50% [46, 55] | 27% [23, 31] | 11% [8, 14] | 3% [2, 5] | 1% [0, 2] | 0% [0, 0] | 8% [6, 11] | 8% [6, 10] | 7% [5, 10] |
| 205->130 | dense | 105 | 50% [45, 55] | 26% [21, 30] | 10% [7, 13] | 2% [1, 4] | 1% [0, 2] | 0% [0, 0] | 9% [6, 12] | 7% [5, 10] | 8% [6, 10] |
| 141->161 | dense | 105 | 41% [35, 47] | 17% [13, 22] | 4% [2, 6] | 1% [0, 2] | 1% [0, 1] | 0% [0, 0] | 3% [2, 5] | 2% [1, 4] | 3% [1, 5] |
| 39->76 | dense | 105 | 57% [52, 62] | 33% [27, 39] | 16% [13, 19] | 6% [4, 8] | 2% [1, 3] | 0% [0, 0] | 8% [6, 11] | 7% [5, 10] | 7% [5, 10] |
| 159->162 | sparse | 15 | 45% [18, 72] | 28% [11, 47] | 13% [5, 27] | 1% [0, 5] | 0% [0, 0] | 0% [0, 0] | 5% [1, 11] | 1% [0, 5] | 1% [0, 5] |
| 83->130 | sparse | 20 | 41% [23, 59] | 14% [3, 29] | 6% [0, 15] | 1% [0, 4] | 0% [0, 0] | 0% [0, 0] | 5% [0, 14] | 1% [0, 4] | 1% [0, 4] |
| 26->76 | sparse | 17 | 55% [31, 76] | 38% [15, 58] | 28% [9, 44] | 19% [4, 32] | 5% [0, 11] | 0% [0, 0] | 29% [12, 46] | 7% [1, 14] | 21% [5, 36] |
| 152->68 | sparse | 14 | 36% [6, 58] | 23% [2, 43] | 13% [0, 29] | 6% [0, 13] | 1% [0, 4] | 0% [0, 0] | 16% [0, 32] | 13% [0, 29] | 6% [0, 13] |
| 204->162 | sparse | 11 | 40% [12, 71] | 35% [8, 66] | 35% [8, 66] | 33% [7, 62] | 20% [2, 43] | 0% [0, 0] | 27% [2, 57] | 31% [5, 62] | 35% [8, 66] |
| 235->161 | sparse | 12 | 75% [53, 92] | 43% [24, 66] | 25% [6, 52] | 18% [0, 47] | 8% [0, 23] | 2% [0, 6] | 25% [6, 52] | 18% [3, 40] | 17% [2, 38] |

Spread across routes (weights = kept scenarios per route). MAD10 = weighted mean absolute distance of the route late rates from 10%, in points. The fixed buffer whose overall late rate is closest to 10% is gut B=10.

| strategy | min | max | weighted SD [95% CI] | MAD10 [95% CI] |
|---|---|---|---|---|
| gut B=0 | 36% (152->68) | 75% (235->161) | 6.3 [5.1, 10.0] | 39.4 [36.9, 42.3] |
| gut B=5 | 14% (83->130) | 43% (235->161) | 5.9 [5.3, 8.9] | 15.3 [12.7, 18.7] |
| gut B=10 | 4% (141->161) | 35% (204->162) | 6.0 [4.6, 9.3] | 4.1 [3.6, 5.5] |
| gut B=15 | 1% (10->130) | 33% (204->162) | 5.1 [3.1, 8.4] | 7.7 [7.2, 8.4] |
| gut B=20 | 0% (10->130) | 20% (204->162) | 2.6 [1.0, 5.4] | 9.1 [8.9, 9.4] |
| gut B=30 | 0% (215->130) | 2% (235->161) | 0.2 [0.0, 0.7] | 10.0 [9.9, 10.0] |
| lookup p>=0.90 | 3% (141->161) | 29% (26->76) | 5.2 [3.4, 8.6] | 4.0 [3.3, 5.1] |
| LightGBM p>=0.90 | 1% (83->130) | 31% (204->162) | 4.1 [2.3, 7.6] | 4.2 [3.5, 5.3] |
| TabPFN p>=0.90 | 1% (83->130) | 35% (204->162) | 4.8 [3.0, 8.0] | 4.6 [3.8, 5.5] |

Model strategies minus gut B=10 (paired over the same bootstrap draws):

- lookup p>=0.90: weighted SD -0.8 [-2.3, 0.8], MAD10 -0.1 [-1.4, 0.5] points
- LightGBM p>=0.90: weighted SD -1.9 [-3.4, -0.8], MAD10 0.1 [-1.4, 1.1] points
- TabPFN p>=0.90: weighted SD -1.2 [-2.5, -0.4], MAD10 0.5 [-1.2, 1.3] points

## Method notes

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

