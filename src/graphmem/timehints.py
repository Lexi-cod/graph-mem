"""Deterministic resolution of relative time expressions against a session date.

The extractor model is unreliable at date arithmetic ("last year" was dropped rather
than converted), so we resolve common expressions here and hand the model the
answers as hints. Output phrasing follows LoCoMo's answer style, e.g.
"the week before 8 May 2023", "2022", "7 May 2023".
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "a couple": 2}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_SEASONS = {"spring": (3, 5), "summer": (6, 8), "fall": (9, 11), "autumn": (9, 11), "winter": (12, 2)}


def _d(x: date) -> str:
    return f"{x.day} {x:%B %Y}"


def _month_shift(x: date, n: int) -> date:
    m = x.month - 1 + n
    return date(x.year + m // 12, m % 12 + 1, 1)


def _season(name: str, today: date, which: str) -> str:
    start_m, _ = _SEASONS[name]
    label = "fall" if name == "autumn" else name

    def start(y: int) -> date:
        return date(y, start_m, 1)

    def end(y: int) -> date:  # first day after the season (winter spans the new year)
        e = _month_shift(start(y), 3)
        return e

    def name_of(y: int) -> str:
        return f"winter {y}-{str(y + 1)[2:]}" if label == "winter" else f"{label} {y}"

    if which == "last":  # most recent occurrence that has ended
        y = today.year
        while end(y) > today:
            y -= 1
        return name_of(y)
    if which == "next":  # next occurrence that has not started
        y = today.year - 1
        while start(y) <= today:
            y += 1
        return name_of(y)
    # "this": the current occurrence, else the next one
    for y in (today.year - 1, today.year, today.year + 1):
        if start(y) <= today < end(y) or start(y) > today:
            return name_of(y)
    return name_of(today.year)


def _rules(today: date) -> list[tuple[str, callable, object]]:
    wd = today.weekday()
    num = r"(\d+|a couple of|a couple|an?|one|two|three|four|five|six|seven|eight|nine|ten)"

    def n_of(s: str) -> int:
        return int(s) if s.isdigit() else _NUM[s]

    def ago(m: re.Match) -> str:
        n, unit = n_of(m.group(1).lower()), m.group(2).lower()
        if unit.startswith("day"):
            return _d(today - timedelta(days=n))
        if unit.startswith("week"):
            return f"{'about ' if n > 1 else ''}{'the week' if n == 1 else f'{n} weeks'} before {_d(today)}"
        if unit.startswith("month"):
            return f"{_month_shift(today, -n):%B %Y}"
        return str(today.year - n)

    def weekday(m: re.Match) -> str:
        which, day = m.group(1).lower(), _WEEKDAYS.index(m.group(2).lower())
        if which == "last":
            delta = (wd - day) % 7 or 7
            return f"{m.group(2).capitalize()} {_d(today - timedelta(days=delta))}"
        delta = (day - wd) % 7
        if which == "next" and delta == 0:
            delta = 7
        return f"{m.group(2).capitalize()} {_d(today + timedelta(days=delta))}"

    def ago_prep(m: re.Match) -> str:
        u = m.group(2).lower()
        return "on" if u.startswith("day") else ("" if u.startswith("week") else "in")

    # (pattern, value, preposition used when the value is substituted inline)
    return [
        (r"the day before yesterday", lambda m: _d(today - timedelta(days=2)), "on"),
        (r"yesterday", lambda m: _d(today - timedelta(days=1)), "on"),
        (r"last night", lambda m: _d(today - timedelta(days=1)), "on the night of"),
        (r"the day after tomorrow", lambda m: _d(today + timedelta(days=2)), "on"),
        (r"tomorrow", lambda m: _d(today + timedelta(days=1)), "on"),
        (r"today|tonight|this (?:morning|afternoon|evening)", lambda m: _d(today), "on"),
        (rf"{num} (days?|weeks?|months?|years?) ago", ago, ago_prep),
        (r"a few days ago", lambda m: f"a few days before {_d(today)}", ""),
        (r"a few weeks ago", lambda m: f"a few weeks before {_d(today)}", ""),
        (r"a few months ago", lambda m: f"a few months before {today:%B %Y}", ""),
        (r"a few years ago", lambda m: f"a few years before {today.year}", ""),
        (r"last weekend", lambda m: f"the weekend before {_d(today)}", ""),
        (r"this weekend", lambda m: f"the weekend of {_d(today)}", ""),
        (r"next weekend", lambda m: f"the weekend after {_d(today)}", ""),
        (r"last week", lambda m: f"the week before {_d(today)}", ""),
        (r"this week", lambda m: f"the week of {_d(today)}", ""),
        (r"next week", lambda m: f"the week after {_d(today)}", ""),
        (r"last month", lambda m: f"{_month_shift(today, -1):%B %Y}", "in"),
        (r"this month", lambda m: f"{today:%B %Y}", "in"),
        (r"next month", lambda m: f"{_month_shift(today, 1):%B %Y}", "in"),
        (r"last year", lambda m: str(today.year - 1), "in"),
        (r"this year", lambda m: str(today.year), "in"),
        (r"next year", lambda m: str(today.year + 1), "in"),
        (r"(last|this|next) (spring|summer|fall|autumn|winter)",
         lambda m: _season(m.group(2).lower(), today, m.group(1).lower()), "in"),
        (rf"(last|this|next) ({'|'.join(_WEEKDAYS)})", weekday, "on"),
    ]


def _matches(text: str, session_time: datetime) -> list[tuple[int, int, str, str]]:
    """Non-overlapping (start, end, value, inline phrase), in order of appearance."""
    today = session_time.date()
    found: list[tuple[int, int, str, str]] = []
    for pattern, fn, prep in _rules(today):
        for m in re.finditer(rf"\b(?:{pattern})\b", text, flags=re.IGNORECASE):
            if any(a < m.end() and m.start() < b for a, b, _, _ in found):
                continue  # overlaps a longer/earlier rule ("the day before yesterday")
            val = fn(m)
            p = prep(m) if callable(prep) else prep
            found.append((m.start(), m.end(), val, f"{p} {val}".strip()))
    return sorted(found)


def time_hints(text: str, session_time: datetime) -> list[tuple[str, str]]:
    """[(expression as written, resolved value)], in order of appearance, deduplicated."""
    seen, out = set(), []
    for a, b, val, _ in _matches(text, session_time):
        expr = text[a:b]
        if expr.lower() not in seen:
            seen.add(expr.lower())
            out.append((expr, val))
    return out


_PREP_BEFORE = re.compile(r"\b(in|on|since|from|during|until|till|by|before|after|of|for|at)\s+$",
                          re.IGNORECASE)


def rewrite_relative(text: str, session_time: datetime) -> tuple[str, int]:
    """Replace relative time expressions in `text` with absolute ones, keeping the
    sentence readable ("painted it last year" -> "painted it in 2022"; "since last
    year" -> "since 2022"). Returns (new text, number of replacements)."""
    out, pos, n = [], 0, 0
    for a, b, val, inline in _matches(text, session_time):
        before = text[pos:a]
        # bare value after a preposition or an opening parenthesis ("trip (last week)")
        phrase = val if _PREP_BEFORE.search(before) or before.rstrip().endswith("(") else inline
        out.append(before + phrase)
        pos, n = b, n + 1
    out.append(text[pos:])
    return "".join(out), n


# ---------------------------------------------------------------------------- date intervals

_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december"]
_MON = "|".join(_MONTH_NAMES)
_DAY_MONTH_YEAR = rf"(\d{{1,2}})\s+({_MON})\s+(\d{{4}})"


def _dmy(d: str, m: str, y: str) -> date:
    return date(int(y), _MONTH_NAMES.index(m.lower()) + 1, int(d))


def _month_span(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), _month_shift(date(y, m, 1), 1) - timedelta(days=1)


def _interval_rules() -> list[tuple[str, callable]]:
    def rel(before: int, after: int):
        # window relative to an anchor date: [anchor+before, anchor+after]
        return lambda m: (_dmy(*m.group(1, 2, 3)) + timedelta(days=before),
                          _dmy(*m.group(1, 2, 3)) + timedelta(days=after))

    def few_months(m):
        y, mo = int(m.group(2)), _MONTH_NAMES.index(m.group(1).lower()) + 1
        a = _month_shift(date(y, mo, 1), -6)
        return a, date(y, mo, 1) - timedelta(days=1)

    def n_weeks(m):
        n, anchor = int(m.group(1)), _dmy(*m.group(2, 3, 4))
        return anchor - timedelta(days=7 * (n + 1)), anchor - timedelta(days=7 * (n - 1))

    def day_range(m):
        y, mo = int(m.group(4)), _MONTH_NAMES.index(m.group(3).lower()) + 1
        return date(y, mo, int(m.group(1))), date(y, mo, int(m.group(2)))

    def month(m):
        return _month_span(int(m.group(2)), _MONTH_NAMES.index(m.group(1).lower()) + 1)

    def season(m):
        name, y = m.group(1).lower(), int(m.group(2))
        start_m = _SEASONS[name][0]
        a = date(y, start_m, 1)
        return a, _month_shift(a, 3) - timedelta(days=1)

    def year(m):
        y = int(m.group(0))
        return date(y, 1, 1), date(y, 12, 31)

    return [
        (rf"week before {_DAY_MONTH_YEAR}", rel(-7, -1)),
        (rf"week after {_DAY_MONTH_YEAR}", rel(1, 7)),
        (rf"week of {_DAY_MONTH_YEAR}", rel(-3, 3)),
        (rf"weekend before {_DAY_MONTH_YEAR}", rel(-8, -1)),
        (rf"weekend after {_DAY_MONTH_YEAR}", rel(1, 8)),
        (rf"weekend of {_DAY_MONTH_YEAR}", rel(-2, 2)),
        (rf"few days before {_DAY_MONTH_YEAR}", rel(-7, -1)),
        (rf"few weeks before {_DAY_MONTH_YEAR}", rel(-42, -7)),
        (rf"(\d+) weeks before {_DAY_MONTH_YEAR}", n_weeks),
        (rf"few months before ({_MON})\s+(\d{{4}})", few_months),
        (rf"(\d{{1,2}})\s*[-–]\s*(\d{{1,2}})\s+({_MON})\s+(\d{{4}})", day_range),
        (_DAY_MONTH_YEAR, rel(0, 0)),
        (rf"\b({_MON})\s+(\d{{4}})", month),
        (r"\b(spring|summer|fall|autumn|winter)\s+(\d{4})", season),
        (r"\b(?:19|20)\d{2}\b", year),
    ]


def name_interval(text: str) -> tuple[date, date] | None:
    """The date interval an entity name refers to (hull of all date mentions), or None.
    "week before 23 August 2023" -> 16..22 Aug; "August 2023" -> 1..31 Aug."""
    spans: list[tuple[int, int]] = []
    found: list[tuple[date, date]] = []
    for pattern, fn in _interval_rules():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if any(a < m.end() and m.start() < b for a, b in spans):
                continue
            try:
                found.append(fn(m))
            except ValueError:  # e.g. 31 February
                continue
            spans.append((m.start(), m.end()))
    if not found:
        return None
    return min(a for a, _ in found), max(b for _, b in found)


def intervals_compatible(a: str, b: str) -> bool:
    """True if either name is undated or their date intervals overlap."""
    ia, ib = name_interval(a), name_interval(b)
    return ia is None or ib is None or (ia[0] <= ib[1] and ib[0] <= ia[1])
