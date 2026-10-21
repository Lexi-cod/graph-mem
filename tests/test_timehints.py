from datetime import datetime, timezone

import pytest

from graphmem.timehints import time_hints

MON = datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)  # Monday 8 May 2023


def hints(text, t=MON):
    return dict(time_hints(text, t))


@pytest.mark.parametrize("text,expr,want", [
    ("I went to a LGBTQ support group yesterday", "yesterday", "7 May 2023"),
    ("I painted that lake sunrise last year!", "last year", "2022"),
    ("we went camping last week", "last week", "the week before 8 May 2023"),
    ("see you next month", "next month", "June 2023"),
    ("it was 3 days ago", "3 days ago", "5 May 2023"),
    ("two years ago I moved", "two years ago", "2021"),
    ("a couple of months ago", "a couple of months ago", "March 2023"),
    ("last Sunday was fun", "last Sunday", "Sunday 7 May 2023"),
    ("last Monday", "last Monday", "Monday 1 May 2023"),  # same weekday -> a week ago
    ("next Friday", "next Friday", "Friday 12 May 2023"),
    ("last summer we hiked", "last summer", "summer 2022"),
    ("this summer we will hike", "this summer", "summer 2023"),
    ("last winter was cold", "last winter", "winter 2022-23"),
    ("the day before yesterday", "the day before yesterday", "6 May 2023"),
    ("a few weeks ago", "a few weeks ago", "a few weeks before 8 May 2023"),
])
def test_resolves(text, expr, want):
    assert hints(text)[expr] == want


def test_no_hints_and_no_partial_overlaps():
    assert time_hints("I love painting sunsets", MON) == []
    h = time_hints("the day before yesterday", MON)
    assert h == [("the day before yesterday", "6 May 2023")]  # not also "yesterday"


def test_order_and_dedupe():
    h = time_hints("Yesterday was great, and last year too. yesterday!", MON)
    assert h == [("Yesterday", "7 May 2023"), ("last year", "2022")]


def test_last_summer_during_summer_is_previous_year():
    assert hints("last summer", datetime(2023, 7, 1, tzinfo=timezone.utc))["last summer"] == "summer 2022"
    assert hints("last summer", datetime(2023, 10, 1, tzinfo=timezone.utc))["last summer"] == "summer 2023"


@pytest.mark.parametrize("text,want", [
    ("Melanie painted a lake sunrise last year.", "Melanie painted a lake sunrise in 2022."),
    ("Caroline went to a support group yesterday.", "Caroline went to a support group on 7 May 2023."),
    ("She has painted since last year.", "She has painted since 2022."),
    ("They went camping last week.", "They went camping the week before 8 May 2023."),
    ("Jon opened it 3 days ago.", "Jon opened it on 5 May 2023."),
    ("Gina plans a trip next month.", "Gina plans a trip in June 2023."),
    ("No relative time here (7 May 2023).", "No relative time here (7 May 2023)."),
])
def test_rewrite_relative(text, want):
    from graphmem.timehints import rewrite_relative
    assert rewrite_relative(text, MON)[0] == want
