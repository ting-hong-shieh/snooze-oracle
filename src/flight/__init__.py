"""Flight workstream: ArrDelay samples and cancellation probability for JFK/LGA arrivals."""

__all__ = ["flight_arrival_samples", "p_cancel"]


def __getattr__(name):  # lazy, so `python -m src.flight.evaluate` does not import it twice
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
