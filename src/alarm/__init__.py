"""Pickup-scenario alarm (SPEC "Alarm"): reads precomputed tables from data/alarm/."""

from .core import latest_safe_wake, p_on_time, snooze_curve

__all__ = ["p_on_time", "snooze_curve", "latest_safe_wake"]
