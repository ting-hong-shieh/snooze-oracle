# Contributing

Thanks for looking. Snooze Oracle is a small research project, so issues and pull requests
are welcome, and small ones are easiest to review.

## Reporting a problem

Open an issue with:

- what you ran (the command, or the demo scenario and buttons you pressed);
- what you expected and what happened;
- your Python version and OS, and your GPU if it is a TabPFN step.

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep one change per pull request, and say in the description why it is needed.
3. If a change moves a number in a report, rerun the step that makes it and update the
   report in the same pull request. [SPEC.md](SPEC.md) defines every number; update it if a
   definition changes.
4. Data files and model weights stay out of git (`data/` and `out/` are ignored).

## Good first contributions

- More commute routes, or another city with open trip records.
- A phone alarm that enforces the snooze cutoff (today the refusal exists only on the demo
  page).
- Better predictions for short early flights into LaGuardia, where the airport alarm gives
  too much extra sleep (see [docs/backtest.md](docs/backtest.md)).

## Licences

Code contributions are under Apache-2.0, like the rest of the code. TabPFN-3.5's weights have
their own non-commercial license from Prior Labs; don't add them to the repo.
