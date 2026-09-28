"""``weather`` -- mocked forecast.

Deterministic on purpose: same city + same day always yields the same numbers,
so a test can assert on the agent's reasoning about the result, and a demo
replays identically on both windows.
"""

from __future__ import annotations

import hashlib
from datetime import date as Date
from datetime import timedelta
from typing import Annotated

from ..errors import ToolError
from .base import tool

# Rough climate normals, (Jan..Dec) mean high / mean low in Celsius. Enough for
# the model to reason about seasonality, which is all this tool is for.
_NORMALS: dict[str, tuple[tuple[int, int], ...]] = {
    "default": ((9, 2), (11, 3), (16, 7), (22, 12), (27, 17), (30, 21),
                (33, 25), (33, 25), (28, 20), (23, 15), (17, 9), (12, 4)),
    "shanghai": ((9, 3), (11, 4), (15, 8), (21, 13), (26, 18), (29, 22),
                 (32, 25), (32, 25), (28, 21), (23, 16), (18, 10), (13, 5)),
    "beijing": ((-2, -9), (2, -6), (9, 0), (19, 7), (26, 13), (30, 19),
                (32, 23), (31, 22), (26, 15), (18, 7), (9, -1), (1, -7)),
    "shenzhen": ((20, 12), (21, 13), (23, 16), (27, 20), (30, 24), (31, 26),
                 (32, 26), (32, 26), (31, 25), (29, 22), (26, 18), (22, 13)),
    "london": ((7, 2), (8, 2), (11, 4), (14, 6), (18, 9), (21, 12),
               (23, 14), (23, 14), (19, 11), (15, 9), (10, 5), (7, 3)),
    "new york": ((4, -2), (6, -1), (10, 3), (17, 7), (22, 13), (27, 18),
                 (30, 22), (29, 21), (25, 17), (18, 11), (12, 5), (7, 0)),
    "tokyo": ((10, 2), (11, 3), (14, 6), (19, 11), (23, 16), (26, 20),
              (30, 23), (31, 24), (27, 21), (22, 15), (17, 9), (12, 4)),
}

_SKY = ("clear", "partly cloudy", "overcast", "light rain", "rain", "thunderstorm", "snow")


def _stable_rng(city: str, day: Date) -> float:
    """A deterministic 0..1 value for (city, day) -- no global random state."""
    digest = hashlib.sha256(f"{city.strip().lower()}|{day.isoformat()}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _key(city: str) -> tuple[tuple[int, int], ...]:
    needle = city.strip().lower()
    for name, normals in _NORMALS.items():
        if name != "default" and name in needle:
            return normals
    return _NORMALS["default"]


def _forecast(city: str, day: Date) -> dict[str, object]:
    normals = _key(city)
    hi_n, lo_n = normals[day.month - 1]
    r = _stable_rng(city, day)
    hi = round(hi_n + (r - 0.5) * 6, 1)
    lo = round(lo_n + (r - 0.5) * 5, 1)
    sky = _SKY[int(r * len(_SKY)) % len(_SKY)]
    humidity = int(35 + r * 50)
    wind = round(2 + r * 20, 1)
    return {
        "city": city.strip(),
        "date": day.isoformat(),
        "temp_high_c": hi,
        "temp_low_c": lo,
        "temp_high_f": round(hi * 9 / 5 + 32, 1),
        "temp_low_f": round(lo * 9 / 5 + 32, 1),
        "condition": sky,
        "humidity_pct": humidity,
        "wind_kph": wind,
        "source": "mock-forecast (deterministic; not a real weather service)",
    }


@tool(tags=("knowledge",))
def weather(
    city: Annotated[str, "city name, e.g. 'Shanghai' or 'Berlin'"],
    date: Annotated[str, "YYYY-MM-DD; defaults to today"] = "",
) -> str:
    """Get the current or forecast weather for a city. Use this whenever the user asks about weather, temperature or whether to carry an umbrella."""
    if not city.strip():
        raise ToolError("City is required")
    if date:
        try:
            day = Date.fromisoformat(date)  # noqa: A001 - `date` is the tool argument
        except ValueError as exc:
            raise ToolError(
                f"Bad date {date!r}",
                hint="Use YYYY-MM-DD, e.g. 2026-04-01.",
            ) from exc
    else:
        day = Date.today()
    return _fmt(_forecast(city, day))


def forecast_range(city: str, start: Date, days: int) -> str:
    """Multi-day view; used by the tests and by the loop's todo reminders."""
    rows = []
    for i in range(days):
        day = start + timedelta(days=i)
        f = _forecast(city, day)
        rows.append(
            f"{f['date']}  {f['temp_low_c']:>5}C ~ {f['temp_high_c']:<5}C  "
            f"{f['condition']:<13} humidity {f['humidity_pct']:>3}%  wind {f['wind_kph']:>4} kph"
        )
    return "\n".join(rows)


def _fmt(f: dict[str, object]) -> str:
    return (
        f"Weather for {f['city']} on {f['date']} (mocked):\n"
        f"  {f['condition']}, {f['temp_low_c']}C to {f['temp_high_c']}C "
        f"({f['temp_low_f']}F to {f['temp_high_f']}F)\n"
        f"  humidity {f['humidity_pct']}%, wind {f['wind_kph']} kph\n"
        f"  source: {f['source']}"
    )