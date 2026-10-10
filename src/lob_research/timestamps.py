"""Integer milliseconds, explicit source-day timezone, source-row ordering."""

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def day_epoch_ns(day: str, zone="America/New_York"):
    d = date.fromisoformat(day)
    midnight = datetime.combine(d, time(), ZoneInfo(zone))
    tomorrow = datetime.combine(d + timedelta(days=1), time(), ZoneInfo(zone))
    if midnight.utcoffset() != tomorrow.utcoffset():
        raise ValueError("DST transition day is ambiguous for wall-clock offsets; unsupported")
    return int(midnight.timestamp()) * 1000000000


def utc_ns(day, milliseconds, zone="America/New_York"):
    if not 0 <= milliseconds < 86400000:
        raise ValueError("source timestamp outside day")
    return day_epoch_ns(day, zone) + milliseconds * 1000000


def iso_utc(ns):
    return (
        datetime.fromtimestamp(ns // 1000000000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        + f".{ns % 1000000000:09d}Z"
    )


def clock_ms(value):
    h, m, s = map(int, value.split(":"))
    if not (0 <= h <= 23 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError("invalid session clock")
    return (h * 3600 + m * 60 + s) * 1000


class Ordering:
    def __init__(self):
        self.previous_row = 0
        self.previous_ms = None

    def observe(self, event):
        if event.row != self.previous_row + 1:
            raise ValueError("noncontiguous source row; do not sort or drop input")
        regressed = self.previous_ms is not None and event.time_ms < self.previous_ms
        duplicate = self.previous_ms is not None and event.time_ms == self.previous_ms
        self.previous_row, self.previous_ms = event.row, event.time_ms
        return regressed, duplicate
