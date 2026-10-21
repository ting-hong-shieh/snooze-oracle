# Findings

## Snoozing 10 minutes does not save money; only hour-level effects around 6am exist

Data: Uber trips, August 2026 (`python scripts/explore.py 2026-08`, section 4).
Routes: the 5 busiest routes to an airport (`do_zone` in {1, 132, 138}, `pu_zone <> do_zone`),
ranked by trips requested 03:00–06:59, all days of the week. Fare = `base_passenger_fare`.

| Route | Trips 03:00–06:59 | Mean change between adjacent 10-min slots | Mean P90–P10 within a slot | Hourly median range (0–23h) |
|---|---|---|---|---|
| 216 South Ozone Park → JFK | 2,532 | $1.09 | $19.58 | $10.82–22.32 |
| 230 Times Sq/Theatre District → LGA | 1,878 | $2.41 | $34.43 | $34.34–74.81 |
| 10 Baisley Park → JFK | 1,829 | $0.86 | $15.10 | $13.58–21.14 |
| 7 Astoria → LGA | 1,524 | $0.82 | $11.75 | $12.67–22.44 |
| 129 Jackson Heights → LGA | 1,476 | $1.10 | $12.55 | $9.08–17.97 |

- Between adjacent 10-minute request slots (03:00–06:59), the median fare moves by about
  $1–2, 10–15 times less than the spread of fares inside a single slot. The 03:00–03:30
  slots hold only 9–31 trips, so their medians are mostly noise.
- Fare per mile moves by $0.13–0.31 between adjacent slots.
- By hour, fares rise from 3am to 6am. Times Sq → LGA: median $49.87 at 3am, $59.96 at 6am;
  fare per mile $4.93 → $5.93, so the rise is not only longer routes.
- On both JFK routes the 6am hour is the most expensive early-morning hour: 216 → JFK
  $22.32 at 6am vs $20.12 at 7am; 10 → JFK $21.14 at 6am vs $17.78 at 7am.
- Decision: drop "wait 10 minutes to save on the fare" as a product feature
  (see `SPEC.md`). Keep `explore.py` section 4 so the check can be rerun on other months.

Earlier check, before the route definition changed: the busiest weekday 6–10am route was
76 → 76 (inside East New York, about 1.2 miles); slot medians ranged $8.79–9.43.

## Previous-day delay features help P90 coverage but not pinball, and cannot be deployed

Data: flights into JFK/LGA scheduled 03:00–08:59, train 2025-08 … 2026-07, test 2026-08
(`src/flight/RESULTS_round2.md`). Features: the origin airport's mean departure delay and
the destination airport's mean arrival delay on day A−2 (A = scheduled arrival date).

- P90 coverage moved toward 0.90: TabPFN 0.854 → 0.881 (paired +0.027 [+0.013, +0.046]),
  LightGBM 0.820 → 0.842 (+0.021 [+0.006, +0.037]).
- Mean pinball did not improve: TabPFN −0.010 [−0.214, +0.263], LightGBM −0.094
  [−0.262, +0.070].
- The features do not separate bad days from calm ones (observed P90 delay 52 / 48 / 50 min
  after calm / middle / bad A−2 days). They carry the month's level, which happened to match
  August 2026.
- TabPFN copied extreme days: an A−2 value can act as a key for one training day. On
  2026-08-23 JFK's value matched 2026-02-24 (a storm-recovery morning) and TabPFN's predicted
  P90 had a median of 486 min across 56 flights; the actual P90 was 101 min.
- Not deployable as built: BTS publishes about two months late, so a real product would need
  live flight status for any previous-day feature.
- Decision: the runtime default is the "dest only" variant (adding `dest` alone gave the only
  significant pinball gain this round, TabPFN −0.030 [−0.048, −0.013]).
