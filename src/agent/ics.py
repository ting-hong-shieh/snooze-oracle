"""A calendar event with two alarms: first ring 30 min before the wake time, final at it."""

from __future__ import annotations

from datetime import datetime, timezone


def _fmt(t: datetime) -> str:
    return t.strftime("%Y%m%dT%H%M%S")


def make(wake: datetime, summary: str, description: str, first_ring_min: int = 30) -> str:
    """Floating local times (no time zone), so the alarm rings at that clock time on the
    device that imports it."""
    uid = f"{_fmt(wake)}-snooze-oracle"
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    desc = description.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,") \
        .replace("\n", "\\n")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Snooze Oracle//EN", "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{now}", f"DTSTART:{_fmt(wake)}",
        f"DTEND:{_fmt(wake)}", f"SUMMARY:{summary}", f"DESCRIPTION:{desc}",
        "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:First ring: {summary}",
        f"TRIGGER:-PT{first_ring_min}M", "END:VALARM",
        "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:Final alarm: {summary}",
        "TRIGGER:PT0M", "END:VALARM",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"
