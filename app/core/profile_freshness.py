"""When a Gemini profile is old enough to be worth paying for again.

Two surfaces answer this question. The UI's "Estimated New Gemini Calls"
counted rows with neither `gemini_insights` nor `gemini_insights_structured`;
the enrichment the Predict button triggers looked at the structured column
alone. They agreed only because the legacy column is NULL on every row -- an
accident, not a design. That is D1, and the fix is not a better estimate but a
single predicate both call.

Staleness is new here. `gemini_profiled_at` was backfilled to the migration
timestamp for every row that already had a profile (Phase 5, following
`scripts/migrate_predicted_at.py`), so nothing in production reads as stale
until GEMINI_PROFILE_MAX_AGE_DAYS after 2026-09-23. Turning it on costs
nothing today; the first sweep is a deliberate, budget-capped decision rather
than something that happens the moment this lands.
"""
import datetime
from typing import Any, Iterable, Mapping, Optional

# Six months. A profile describes a restaurant's cuisine, menu language and
# neighbourhood -- things that move on the scale of a refurbishment, not a
# week. Shortening this multiplies the Gemini bill by the same factor, so it is
# a budget decision as much as a quality one.
GEMINI_PROFILE_MAX_AGE_DAYS = 180
MAPS_LOOKUP_MAX_AGE_DAYS = 60

PROFILE_COLUMN = 'gemini_insights_structured'
PROFILED_AT_COLUMN = 'gemini_profiled_at'
MAPS_LOOKUP_AT_COLUMN = 'maps_lookup_at'

_UTC = datetime.timezone.utc


def _is_missing(value: Any) -> bool:
    """None, NaN or NaT, without importing pandas: neither equals itself."""
    return value is None or value != value


def _as_utc(value: Any) -> Optional[datetime.datetime]:
    """A timestamp from BigQuery, a DataFrame or a JSON string, as aware UTC.

    Returns None for anything unreadable. Every caller treats None as "leave it
    alone", so a parse failure costs nothing; guessing the other way would
    re-profile the table.
    """
    if _is_missing(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = datetime.datetime.fromisoformat(stripped.replace('Z', '+00:00'))
        except ValueError:
            return None
    elif isinstance(value, datetime.datetime):  # pandas Timestamp is a subclass
        parsed = value
    elif isinstance(value, datetime.date):
        parsed = datetime.datetime(value.year, value.month, value.day)
    else:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_UTC)


def has_profile(value: Any) -> bool:
    """Whether a `gemini_insights_structured` value is a profile at all.

    A blank string is not one. The column is model output, and an empty
    response has landed there before.
    """
    return not _is_missing(value) and bool(str(value).strip())


def has_maps_lookup(value: Any) -> bool:
    """Whether a `maps_lookup_at` value is a valid lookup timestamp."""
    return _as_utc(value) is not None


def needs_gemini_profile(
    row_has_profile: bool,
    profiled_at: Any,
    *,
    force: bool = False,
    now: Optional[datetime.datetime] = None,
    max_age_days: Optional[int] = GEMINI_PROFILE_MAX_AGE_DAYS,
    cutoff_date: Optional[Any] = None,
) -> bool:
    """Whether this restaurant should be sent to `AI.GENERATE`.

    `max_age_days=None` (with `cutoff_date=None`) disables staleness, leaving
    "has no profile" as the only trigger. Passing `cutoff_date` refreshes any
    profile stamped strictly before that timestamp/date, and `max_age_days`
    refreshes any profile older than `max_age_days` relative to `now`.
    """
    if force:
        return True
    if not row_has_profile:
        return True
    profiled = _as_utc(profiled_at)
    if profiled is None:
        return False
    cutoff = _as_utc(cutoff_date)
    if cutoff is not None and profiled < cutoff:
        return True
    if max_age_days is None:
        return False
    reference = _as_utc(now) or datetime.datetime.now(_UTC)
    return (reference - profiled) > datetime.timedelta(days=max_age_days)


def needs_maps_lookup(
    maps_lookup_at: Any,
    *,
    force: bool = False,
    now: Optional[datetime.datetime] = None,
    max_age_days: Optional[int] = None,
    cutoff_date: Optional[Any] = None,
) -> bool:
    """Whether this restaurant should be queried against Google Places.

    A missing `maps_lookup_at` means Places has never been asked, so it always
    returns True. Once stamped, a row (including one where `maps_found = False`)
    is re-queried when `force=True`, when `maps_lookup_at < cutoff_date`, or
    when its age exceeds `max_age_days`.
    """
    if force:
        return True
    looked_up = _as_utc(maps_lookup_at)
    if looked_up is None:
        return True
    cutoff = _as_utc(cutoff_date)
    if cutoff is not None and looked_up < cutoff:
        return True
    if max_age_days is None:
        return False
    reference = _as_utc(now) or datetime.datetime.now(_UTC)
    return (reference - looked_up) > datetime.timedelta(days=max_age_days)


def row_needs_gemini_profile(row: Mapping[str, Any], **kwargs: Any) -> bool:
    """`needs_gemini_profile` for a row that still carries its column names."""
    return needs_gemini_profile(
        has_profile(row.get(PROFILE_COLUMN)), row.get(PROFILED_AT_COLUMN), **kwargs
    )


def row_needs_maps_lookup(row: Mapping[str, Any], **kwargs: Any) -> bool:
    """`needs_maps_lookup` for a row that still carries its column names."""
    return needs_maps_lookup(row.get(MAPS_LOOKUP_AT_COLUMN), **kwargs)


def count_needing_gemini_profile(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> int:
    """How many of these rows a run would pay for -- the UI's estimate.

    Accepts a DataFrame as well as an iterable of mappings: iterating a
    DataFrame directly yields its column names, which would count silently and
    wrongly.
    """
    if hasattr(rows, 'to_dict'):
        rows = rows.to_dict('records')
    return sum(1 for row in rows if row_needs_gemini_profile(row, **kwargs))


def count_needing_maps_lookup(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> int:
    """How many of these rows a run would query on Google Places."""
    if hasattr(rows, 'to_dict'):
        rows = rows.to_dict('records')
    return sum(1 for row in rows if row_needs_maps_lookup(row, **kwargs))


def summarise_gemini_freshness(
    rows: Iterable[Mapping[str, Any]], **kwargs: Any
) -> dict[str, int]:
    """Break down a batch into missing, stale/forced, and cached Gemini profiles."""
    if hasattr(rows, 'to_dict'):
        rows = rows.to_dict('records')
    row_list = list(rows)
    total = len(row_list)
    missing = 0
    stale = 0
    for row in row_list:
        present = has_profile(row.get(PROFILE_COLUMN))
        if not present:
            missing += 1
        elif needs_gemini_profile(True, row.get(PROFILED_AT_COLUMN), **kwargs):
            stale += 1
    to_refresh = missing + stale
    return {
        'total': total,
        'missing': missing,
        'stale': stale,
        'cached': total - to_refresh,
        'to_refresh': to_refresh,
    }


def summarise_maps_freshness(
    rows: Iterable[Mapping[str, Any]], **kwargs: Any
) -> dict[str, int]:
    """Break down a batch into missing, stale/forced, and cached Maps lookups."""
    if hasattr(rows, 'to_dict'):
        rows = rows.to_dict('records')
    row_list = list(rows)
    total = len(row_list)
    missing = 0
    stale = 0
    for row in row_list:
        looked_up = has_maps_lookup(row.get(MAPS_LOOKUP_AT_COLUMN))
        if not looked_up:
            missing += 1
        elif needs_maps_lookup(row.get(MAPS_LOOKUP_AT_COLUMN), **kwargs):
            stale += 1
    to_refresh = missing + stale
    return {
        'total': total,
        'missing': missing,
        'stale': stale,
        'cached': total - to_refresh,
        'to_refresh': to_refresh,
    }
