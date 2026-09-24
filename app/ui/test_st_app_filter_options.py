"""The sidebar and the filter agree on what the options are called.

`filter_and_sort_restaurants` used to accept about three spellings of each
answer -- `"Has User Rating (Rated)"`, `"User Rated Only"`, `"Has Rating"` --
and the selectbox that feeds it repeated the literal a fourth time. The aliases
were insurance against the widget and the branch drifting apart, paid for with
a filter nobody could read. Sharing one constant removes the need, but only if
something checks that the shared name is still the one the branch compares
against: a rename that updates the constant and forgets the branch would make
every option silently behave like "All".

So these tests do not assert *which* rows come back. They assert that each
option does something -- that it is a name the filter still recognises.
"""
import pandas as pd
import pytest

from app.ui import st_app
from app.ui.st_app import (
    FILTER_ALL,
    MAPS_OPTIONS,
    MATCH_OPTIONS,
    PRED_OPTIONS,
    RATED_OPTIONS,
    SCOPE_ALL,
    SCOPE_OPTIONS,
    SORT_BY_COLUMN,
    SORT_NATURAL,
    SORT_OPTIONS,
    filter_and_sort_restaurants,
)


@pytest.fixture
def df():
    """One row per answer each filter can give, so no option is a no-op for
    want of a matching row."""
    return pd.DataFrame([
        {"fhrsid": "101", "businessname": "Pizza Palace", "in_scope": True,
         "user_rating": 8.0, "predicted_user_rating": 8.5, "match_score": 92.0,
         "maps_found": True, "maps_rating": 4.5, "priority_score": 70.0,
         "distance_km": 1.2, "first_seen": "2026-01-01",
         "gemini_insights_structured": '{"match_score": 92}'},
        {"fhrsid": "102", "businessname": "Burger Barn", "in_scope": False,
         "user_rating": None, "predicted_user_rating": None, "match_score": None,
         "maps_found": False, "maps_rating": None, "priority_score": 20.0,
         "distance_km": 9.9, "first_seen": "2026-02-01",
         "gemini_insights_structured": None},
        {"fhrsid": "103", "businessname": "Coffee Corner", "in_scope": None,
         "user_rating": None, "predicted_user_rating": 6.0, "match_score": None,
         "maps_found": None, "maps_rating": None, "priority_score": 45.0,
         "distance_km": 4.0, "first_seen": "2026-03-01",
         "gemini_insights_structured": None},
    ])


def ids(frame):
    return list(frame["fhrsid"])


def filtered(df, kwarg, value):
    """One filter applied, nothing sorted -- these tests compare membership,
    and the default sort would reorder the frame under them."""
    return filter_and_sort_restaurants(df, sort_by=SORT_NATURAL, **{kwarg: value})


OPTION_GROUPS = [
    ("scope_filter", SCOPE_OPTIONS, SCOPE_ALL),
    ("user_rating_filter", RATED_OPTIONS, FILTER_ALL),
    ("pred_rating_filter", PRED_OPTIONS, FILTER_ALL),
    ("gemini_match_filter", MATCH_OPTIONS, FILTER_ALL),
    ("maps_filter", MAPS_OPTIONS, FILTER_ALL),
]


@pytest.mark.parametrize("kwarg, options, all_value", OPTION_GROUPS)
def test_every_option_narrows_the_frame(df, kwarg, options, all_value):
    """The "All" member of each group is the no-op; every other member must
    return something different from it, or the branch has stopped matching."""
    unfiltered = ids(filtered(df, kwarg, all_value))
    assert unfiltered == ids(df), f"{kwarg}={all_value!r} should filter nothing"

    for option in options:
        if option == all_value:
            continue
        result = ids(filtered(df, kwarg, option))
        assert result != unfiltered, (
            f"{kwarg}={option!r} matched no branch -- it behaves like {all_value!r}")


@pytest.mark.parametrize("kwarg, options, all_value", OPTION_GROUPS)
def test_the_options_of_a_group_partition_it(df, kwarg, options, all_value):
    """Between them the non-"All" options account for every row exactly once.
    A row that no option selects is unreachable from the sidebar."""
    seen = []
    for option in options:
        if option != all_value:
            seen.extend(ids(filtered(df, kwarg, option)))
    assert sorted(seen) == sorted(ids(df)), f"{kwarg} options do not partition the frame"


def test_every_sort_option_is_either_a_column_or_deliberately_not_one(df):
    unmapped = [o for o in SORT_OPTIONS if o not in SORT_BY_COLUMN]
    assert unmapped == [SORT_NATURAL], (
        f"sort options with no column and no reason to lack one: {unmapped}")


@pytest.mark.parametrize("option", [o for o in SORT_OPTIONS if o != SORT_NATURAL])
def test_every_sort_option_sorts_by_its_column(df, option):
    candidates, ascending = SORT_BY_COLUMN[option]
    column = st_app._first_column(df, candidates)
    result = filter_and_sort_restaurants(df, sort_by=option)
    expected = df.sort_values(by=column, ascending=ascending, na_position="last")
    assert ids(result) == ids(expected)


def test_natural_order_leaves_the_bigquery_order_alone(df):
    assert ids(filter_and_sort_restaurants(df, sort_by=SORT_NATURAL)) == ids(df)


def test_an_unknown_sort_key_also_leaves_it_alone(df):
    """Session state can outlive a renamed option. Falling back to the loaded
    order is visible and harmless; raising would take the page down."""
    assert ids(filter_and_sort_restaurants(df, sort_by="Sorted By Vibes")) == ids(df)


def test_missing_values_sort_last_in_both_directions(df):
    """Ascending or descending, an unscored restaurant is neither the best nor
    the worst one."""
    descending = filter_and_sort_restaurants(df, sort_by="Maps Rating (High to Low)")
    assert ids(descending)[-1] in ("102", "103")

    nearest = filter_and_sort_restaurants(
        df.assign(distance_km=[None, 9.9, 4.0]), sort_by="Distance (Nearest First)")
    assert ids(nearest) == ["103", "102", "101"]
