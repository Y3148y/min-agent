"""``weather`` tool with two backends.

* ``mock`` (default)  -- deterministic fake forecast.  Same city + same day
  always yields the same numbers, so tests can assert on the agent's reasoning,
  and no key or network is needed.
* ``wttr.in``         -- a real (and free, key-less) forecast service.  If it is
  unreachable or returns garbage, the tool quietly falls back to the mock so the
  loop never crashes over a side tool.
"""

from __future__ import annotations

import hashlib
import urllib.parse
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta
from typing import Annotated

from ..errors import ToolError
from .base import ToolSpec, tool

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


def _mock_row(city: str, day: Date) -> dict[str, object]:
    """Deterministic fake forecast for (city, day), explicitly labelled."""
    return _forecast(city, day)


def _parse_day(city: str, date: str) -> Date:
    if not city.strip():
        raise ToolError("City is required")
    if date:
        try:
            return Date.fromisoformat(date)  # noqa: A001 - `date` is the tool argument
        except ValueError as exc:
            raise ToolError(
                f"Bad date {date!r}",
                hint="Use YYYY-MM-DD, e.g. 2026-04-01.",
            ) from exc
    return Date.today()


def fetch_wttr_in(city: str, day: Date, timeout: float = 6.0) -> dict[str, object]:
    """Fetch one day's forecast from wttr.in (no key needed) and normalise it
    into the same shape the mock produces."""
    import httpx

    url = f"https://wttr.in/{urllib.parse.quote(city.strip())}?format=j1&lang=zh"
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    days = data.get("weather", []) or []
    entry = next((d for d in days if d.get("date") == day.isoformat()), days[0] if days else None)
    if entry is None:
        raise RuntimeError(f"wttr.in returned no forecast for {city!r}")
    current = (data.get("current_condition") or [{}])[0]
    hourly0 = (entry.get("hourly") or [{}])[0]
    desc = (
        (hourly0.get("weatherDesc") or [{}])[0].get("value")
        or (current.get("weatherDesc") or [{}])[0].get("value")
        or "unknown"
    )
    return {
        "city": city.strip(),
        "date": entry.get("date") or day.isoformat(),
        "temp_high_c": float(entry.get("maxtempC", 0)),
        "temp_low_c": float(entry.get("mintempC", 0)),
        "temp_high_f": float(entry.get("maxtempF", 0)),
        "temp_low_f": float(entry.get("mintempF", 0)),
        "condition": desc,
        "humidity_pct": int(entry.get("avghumidity") or current.get("humidity") or 0),
        "wind_kph": float(entry.get("maxwindspeedKmph") or current.get("windspeedKmph") or 0),
        "source": "wttr.in",
    }


def _weather_impl(backend: str, timeout: float):
    def impl(city: str, date: str = "") -> str:
        day = _parse_day(city, date)
        if backend == "wttr.in":
            try:
                return _fmt(fetch_wttr_in(city, day, timeout=timeout))
            except Exception:  # noqa: BLE001 - forecast is a side tool; never take the session down
                row = _mock_row(city, day)
                row["source"] = "mock-forecast fallback (wttr.in unreachable)"
                return _fmt(row)
        return _fmt(_mock_row(city, day))

    return impl


def make_weather_tool(backend: str = "mock", *, timeout: float = 6.0) -> ToolSpec:
    """A :class:`ToolSpec` that honours the configured backend.

    Reuses the module-level ``weather`` spec's schema and description, so the
    model sees exactly the same tool regardless of backend.
    """
    return ToolSpec(
        name=weather.name,
        description=weather.description,
        input_schema=weather.input_schema,
        fn=_weather_impl(backend, timeout),
        tags=weather.tags,
    )


@tool(tags=("knowledge",))
def weather(
    city: Annotated[str, "city name, e.g. 'Shanghai' or 'Berlin'"],
    date: Annotated[str, "YYYY-MM-DD; defaults to today"] = "",
) -> str:
    """Get the current or forecast weather for a city. Use this whenever the user asks about weather, temperature or whether to carry an umbrella."""
    day = _parse_day(city, date)
    return _fmt(_mock_row(city, day))


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
        f"Weather for {f['city']} on {f['date']} (source: {f['source']}):\n"
        f"  {f['condition']}, {f['temp_low_c']}C to {f['temp_high_c']}C "
        f"({f['temp_low_f']}F to {f['temp_high_f']}F)\n"
        f"  humidity {f['humidity_pct']}%, wind {f['wind_kph']} kph"
    )