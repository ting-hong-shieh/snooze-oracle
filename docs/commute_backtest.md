# Commute backtest, August 2026 weekdays

Wake up, get ready (prep ~ N(15, 5) min), request an Uber, arrive before the meeting. Strategies pick one wake time the night before; reality uses real August trips (see Scenarios for the two definitions). Method, judgment calls and fairness notes are at the end.

The gut-rule curve is computed from August outcomes, so it is an oracle the night-before user cannot know; the models never saw August.

## Routes

Trips counted on weekday requests 06:00-09:59 (July = training month). Eligible routes: origin in another borough, or July median trip distance >= 3 mi; at most 2 dense routes per destination.

| kind | route | origin | origin borough | destination | July median miles | July trips | August trips |
|---|---|---|---|---|---|---|---|
| dense | 231->161 | TriBeCa/Civic Center | Manhattan | Midtown Center | 3.9 | 1031 | 764 |
| dense | 231->162 | TriBeCa/Civic Center | Manhattan | Midtown East | 4.8 | 798 | 605 |
| dense | 13->246 | Battery Park City | Manhattan | West Chelsea/Hudson Yards | 3.0 | 634 | 461 |
| dense | 112->162 | Greenpoint | Brooklyn | Midtown East | 3.9 | 445 | 404 |
| dense | 145->161 | Long Island City/Hunters Point | Queens | Midtown Center | 3.1 | 425 | 414 |
| dense | 236->246 | Upper East Side North | Manhattan | West Chelsea/Hudson Yards | 4.2 | 401 | 301 |
| sparse | 159->162 | Melrose South | Bronx | Midtown East | 6.7 | 25 | 41 |
| sparse | 146->163 | Long Island City/Queens Plaza | Queens | Midtown North | 2.9 | 27 | 36 |
| sparse | 52->163 | Cobble Hill | Brooklyn | Midtown North | 8.8 | 23 | 36 |
| sparse | 74->88 | East Harlem North | Manhattan | Financial District South | 9.5 | 24 | 30 |
| sparse | 118->88 | Heartland Village/Todt Hill | Staten Island | Financial District South | 13.3 | 19 | 19 |
| sparse | 235->161 | University Heights/Morris Heights | Bronx | Midtown Center | 9.8 | 25 | 35 |

Destinations: Midtown Center (161), Midtown East (162), Midtown North (163), Financial District North (87) and South (88), West Chelsea/Hudson Yards (246).

## Scenarios

21 August weekdays x 12 routes x 5 meeting times (08:00-10:00) = 1260 scenarios.

Reality ride, two definitions:

- **dense routes**: real trips on the same route and the same August date, requested within +-15 min of the strategy's request time, else +-30 min.
- **sparse routes (weaker reality)**: real trips on the same route on ANY August weekday, requested within +-15 min. This keeps sparse scenarios from being skipped, but it loses that day's traffic, so sparse late rates describe a typical weekday rather than the actual day.

Skipped because some strategy's request time had no trip in its pool: {'dense': 20, 'sparse': 170}. Kept: {'dense': 610, 'sparse': 460}. Dense reality draws that needed the +-30 min window: 3.1%. Wake ranges with no time reaching 0.90: {'lookup': 0, 'lgbm': 0, 'tabpfn': 0}. 5 reality draws per scenario.

95% intervals: day-level bootstrap (2,000 resamples of the August weekdays).

## all routes (1,070 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.419 [0.400, 0.440] | 1.4 [1.1, 1.8] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.210 [0.195, 0.226] | 6.6 [6.1, 7.0] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.083 [0.075, 0.091] | 11.4 [11.0, 11.8] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.036 [0.031, 0.041] | 16.3 [15.9, 16.8] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.018 [0.015, 0.021] | 21.4 [21.0, 21.7] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.003 [0.002, 0.004] | 31.7 [31.3, 32.1] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.061 [0.055, 0.069] | 12.9 [12.4, 13.2] | -2.1 [-2.2, -2.1] |
| LightGBM p>=0.90 | 0.066 [0.057, 0.076] | 13.0 [12.5, 13.6] | -2.3 [-2.9, -1.6] |
| TabPFN p>=0.90 | 0.051 [0.044, 0.060] | 13.6 [13.0, 14.1] | -2.9 [-3.5, -2.2] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: -0.004 [-0.010, 0.002]
- LightGBM p>=0.90: 0.003 [-0.002, 0.009]
- TabPFN p>=0.90: -0.005 [-0.011, 0.000]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate -0.010 [-0.017, -0.001], median minutes early 0.7 [-0.0, 1.5], wake time -0.8 [-1.4, -0.1] min
- TabPFN minus LightGBM p>=0.90: late rate -0.015 [-0.018, -0.011], median minutes early 0.6 [0.4, 0.8], wake time -0.6 [-0.8, -0.5] min

## dense routes (610 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.411 [0.376, 0.447] | 1.6 [1.0, 2.3] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.185 [0.157, 0.212] | 6.8 [6.0, 7.6] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.056 [0.044, 0.070] | 11.9 [11.3, 12.6] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.015 [0.009, 0.023] | 17.2 [16.5, 17.8] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.005 [0.002, 0.009] | 22.3 [21.5, 23.0] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.000 [0.000, 0.000] | 32.9 [32.2, 33.6] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.045 [0.033, 0.058] | 12.7 [12.1, 13.3] | -0.9 [-1.0, -0.9] |
| LightGBM p>=0.90 | 0.051 [0.040, 0.064] | 12.1 [11.6, 12.6] | -0.2 [-0.8, 0.4] |
| TabPFN p>=0.90 | 0.048 [0.038, 0.060] | 12.4 [11.9, 12.8] | -0.6 [-1.2, 0.1] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: -0.004 [-0.011, 0.004]
- LightGBM p>=0.90: -0.003 [-0.010, 0.002]
- TabPFN p>=0.90: -0.004 [-0.010, 0.002]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate 0.003 [-0.004, 0.010], median minutes early -0.3 [-1.1, 0.4], wake time 0.3 [-0.3, 1.0] min
- TabPFN minus LightGBM p>=0.90: late rate -0.003 [-0.007, 0.000], median minutes early 0.3 [0.1, 0.5], wake time -0.3 [-0.5, -0.2] min

## sparse routes (460 scenarios)

| strategy | late rate | median minutes early | mean wake vs gut B=10 (min) |
|---|---|---|---|
| gut B=0 | 0.430 [0.415, 0.446] | 1.2 [1.0, 1.5] | 10.0 [10.0, 10.0] |
| gut B=5 | 0.244 [0.231, 0.258] | 6.2 [5.8, 6.6] | 5.0 [5.0, 5.0] |
| gut B=10 | 0.119 [0.106, 0.132] | 10.7 [10.3, 11.1] | 0.0 [0.0, 0.0] |
| gut B=15 | 0.063 [0.054, 0.070] | 15.2 [14.9, 15.6] | -5.0 [-5.0, -5.0] |
| gut B=20 | 0.034 [0.027, 0.042] | 20.1 [19.6, 20.5] | -10.0 [-10.0, -10.0] |
| gut B=30 | 0.007 [0.004, 0.010] | 29.7 [29.2, 30.1] | -20.0 [-20.0, -20.0] |
| lookup p>=0.90 | 0.083 [0.073, 0.093] | 13.1 [12.6, 13.5] | -3.7 [-3.9, -3.6] |
| LightGBM p>=0.90 | 0.086 [0.069, 0.104] | 14.7 [13.7, 15.7] | -5.0 [-5.7, -4.2] |
| TabPFN p>=0.90 | 0.056 [0.042, 0.071] | 15.6 [14.5, 16.5] | -6.0 [-6.6, -5.3] |

Gap to the gut-rule curve (late rate minus the curve's late rate at the same median minutes early; curve = gut rules for every B from 0 to 30 min, linear between them; negative = below the curve):

- lookup p>=0.90: -0.002 [-0.010, 0.005]
- LightGBM p>=0.90: 0.019 [0.009, 0.027]
- TabPFN p>=0.90: -0.005 [-0.015, 0.004]

Paired differences (same scenarios and draws):

- TabPFN minus lookup p>=0.90: late rate -0.027 [-0.039, -0.013], median minutes early 2.5 [1.5, 3.4], wake time -2.2 [-2.9, -1.5] min
- TabPFN minus LightGBM p>=0.90: late rate -0.030 [-0.038, -0.023], median minutes early 0.8 [0.3, 1.4], wake time -1.0 [-1.3, -0.7] min

## Risk by route

Late rate per route, 95% day-bootstrap intervals.

| route | kind | scenarios | gut B=0 | gut B=5 | gut B=10 | gut B=15 | gut B=20 | gut B=30 | lookup p>=0.90 | LightGBM p>=0.90 | TabPFN p>=0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 231->161 | dense | 105 | 38% [32, 44] | 17% [13, 21] | 5% [3, 7] | 1% [0, 2] | 0% [0, 1] | 0% [0, 0] | 3% [1, 5] | 3% [2, 5] | 3% [2, 5] |
| 231->162 | dense | 105 | 30% [23, 37] | 13% [9, 17] | 4% [2, 8] | 1% [0, 4] | 1% [0, 2] | 0% [0, 0] | 4% [2, 7] | 6% [3, 9] | 5% [2, 8] |
| 13->246 | dense | 104 | 45% [40, 50] | 19% [15, 24] | 5% [2, 7] | 1% [0, 2] | 0% [0, 1] | 0% [0, 0] | 8% [5, 11] | 7% [4, 9] | 7% [4, 10] |
| 112->162 | dense | 105 | 43% [36, 50] | 23% [17, 30] | 8% [5, 12] | 3% [1, 5] | 1% [0, 2] | 0% [0, 0] | 6% [3, 8] | 5% [3, 7] | 5% [3, 7] |
| 145->161 | dense | 97 | 45% [38, 52] | 19% [13, 26] | 6% [3, 8] | 2% [1, 3] | 1% [0, 2] | 0% [0, 0] | 2% [1, 5] | 3% [1, 5] | 4% [2, 6] |
| 236->246 | dense | 94 | 47% [42, 54] | 20% [15, 25] | 6% [3, 8] | 1% [0, 2] | 0% [0, 1] | 0% [0, 0] | 4% [2, 7] | 7% [4, 11] | 5% [3, 7] |
| 159->162 | sparse | 82 | 38% [35, 42] | 20% [16, 24] | 4% [2, 6] | 1% [0, 2] | 0% [0, 1] | 0% [0, 0] | 3% [2, 5] | 1% [0, 2] | 1% [0, 2] |
| 146->163 | sparse | 69 | 8% [5, 11] | 3% [1, 6] | 2% [1, 3] | 1% [0, 2] | 0% [0, 0] | 0% [0, 0] | 0% [0, 0] | 2% [1, 4] | 2% [1, 3] |
| 52->163 | sparse | 105 | 56% [52, 59] | 33% [29, 36] | 18% [14, 20] | 8% [6, 10] | 3% [2, 4] | 1% [0, 1] | 11% [9, 13] | 17% [13, 22] | 5% [2, 8] |
| 74->88 | sparse | 78 | 58% [54, 63] | 32% [29, 36] | 17% [14, 20] | 9% [6, 11] | 4% [3, 6] | 0% [0, 0] | 8% [5, 11] | 8% [5, 12] | 9% [6, 12] |
| 118->88 | sparse | 63 | 32% [28, 37] | 15% [11, 20] | 6% [4, 9] | 5% [3, 7] | 5% [3, 7] | 3% [1, 5] | 7% [5, 10] | 4% [2, 7] | 5% [3, 7] |
| 235->161 | sparse | 63 | 58% [54, 63] | 39% [35, 43] | 23% [18, 28] | 14% [10, 18] | 9% [7, 11] | 2% [0, 3] | 21% [17, 27] | 15% [11, 18] | 13% [9, 16] |

Spread across routes (weights = kept scenarios per route). MAD10 = weighted mean absolute distance of the route late rates from 10%, in points. The fixed buffer whose overall late rate is closest to 10% is gut B=9.

| strategy | min | max | weighted SD [95% CI] | MAD10 [95% CI] |
|---|---|---|---|---|
| gut B=0 | 8% (146->163) | 58% (74->88) | 12.5 [11.8, 13.9] | 32.2 [30.3, 34.4] |
| gut B=5 | 3% (146->163) | 39% (235->161) | 8.6 [7.8, 10.0] | 11.9 [10.3, 13.6] |
| gut B=10 | 2% (146->163) | 23% (235->161) | 6.0 [5.2, 7.1] | 5.7 [4.9, 6.6] |
| gut B=15 | 1% (231->161) | 14% (235->161) | 3.8 [3.2, 4.5] | 6.9 [6.5, 7.3] |
| gut B=20 | 0% (146->163) | 9% (235->161) | 2.4 [1.9, 3.0] | 8.2 [7.9, 8.5] |
| gut B=30 | 0% (231->161) | 3% (118->88) | 0.7 [0.4, 1.2] | 9.7 [9.6, 9.8] |
| lookup p>=0.90 | 0% (146->163) | 21% (235->161) | 4.7 [4.0, 5.9] | 5.4 [4.7, 6.1] |
| LightGBM p>=0.90 | 1% (159->162) | 17% (52->163) | 4.6 [3.7, 6.1] | 5.4 [4.6, 6.3] |
| TabPFN p>=0.90 | 1% (159->162) | 13% (235->161) | 2.8 [2.4, 3.7] | 5.2 [4.6, 5.9] |
| gut B=9 | 3% (146->163) | 27% (235->161) | 6.5 [5.6, 7.8] | 5.0 [4.2, 6.1] |

Model strategies minus gut B=9 (paired over the same bootstrap draws):

- lookup p>=0.90: weighted SD -1.7 [-2.7, -0.9], MAD10 0.4 [-0.6, 1.0] points
- LightGBM p>=0.90: weighted SD -1.8 [-2.8, -0.9], MAD10 0.4 [-0.7, 1.1] points
- TabPFN p>=0.90: weighted SD -3.7 [-4.6, -2.8], MAD10 0.2 [-1.0, 1.2] points

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

- Destinations are the six zones named in the request (ids from the TLC zone lookup).
- "Residential" origins have no TLC label; they are every zone except the destinations,
  airports, 264/265 and EWR, and must be in another borough than the destination or have a
  July median trip distance of at least 3 miles on that route (06:00-09:59 weekday trips).
- Dense: the most July trips among eligible routes, at most two per destination, taken in
  order of trips. Sparse: 10-29 July trips among eligible routes, one borough at a time
  (boroughs ordered by their best candidate), ranked by August trips, then zone id.
- Weekdays are Monday-Friday; Friday 2026-07-03 (Independence Day observed) stays in training.
- The ride table grid covers request 05:00-10:59; the latest safe wake is searched over the
  4 hours before the meeting.
- The gut-rule curve used for the gap is every B from 0 to 30 minutes; the reported gut
  strategies are B in {0, 5, 10, 15, 20, 30}.
- Risk spread: weights per route = kept scenarios, recomputed in every bootstrap replicate.
  MAD10 (mean distance from 10%) answers "closer to 10%"; the fixed-buffer comparator is the
  gut B whose overall late rate is closest to 10%.

## Fairness

- The gut-rule curve is an oracle: it is drawn from August outcomes; the night-before user
  cannot know which B lands where. Models never saw August.
- Sparse routes use the weaker any-weekday reality, so their late rates are not comparable
  one-to-one with the dense routes' same-date reality.
- Lookup and LightGBM use all July rows in the setting; TabPFN uses a 10k retrieval context
  per route and hour. Nothing was tuned.
- One month of weekdays (21 days) and 12 routes.

