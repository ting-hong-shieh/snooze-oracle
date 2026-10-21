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
