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

PROFILE_COLUMN = 'gemini_insights_structured'
PROFILED_AT_COLUMN = 'gemini_profiled_at'

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
        try:
            parsed = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
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


def needs_gemini_profile(
    row_has_profile: bool,
    profiled_at: Any,
    *,
    force: bool = False,
    now: Optional[datetime.datetime] = None,
    max_age_days: Optional[int] = GEMINI_PROFILE_MAX_AGE_DAYS,
) -> bool:
    """Whether this restaurant should be sent to `AI.GENERATE`.

    `max_age_days=None` disables staleness, leaving "has no profile" as the
    only trigger. That is how the training pre-flight asks: retraining is cheap
    and re-profiling is not, so a scheduled training run fills gaps and never
    refreshes.
    """
    if force:
        return True
    if not row_has_profile:
        return True
    if max_age_days is None:
        return False
    profiled = _as_utc(profiled_at)
    if profiled is None:
        return False
    reference = _as_utc(now) or datetime.datetime.now(_UTC)
    return (reference - profiled) > datetime.timedelta(days=max_age_days)


def row_needs_gemini_profile(row: Mapping[str, Any], **kwargs: Any) -> bool:
    """`needs_gemini_profile` for a row that still carries its column names."""
    return needs_gemini_profile(
        has_profile(row.get(PROFILE_COLUMN)), row.get(PROFILED_AT_COLUMN), **kwargs
    )


def count_needing_gemini_profile(rows: Iterable[Mapping[str, Any]], **kwargs: Any) -> int:
    """How many of these rows a run would pay for -- the UI's estimate.

    Accepts a DataFrame as well as an iterable of mappings: iterating a
    DataFrame directly yields its column names, which would count silently and
    wrongly.
    """
    if hasattr(rows, 'to_dict'):
        rows = rows.to_dict('records')
    return sum(1 for row in rows if row_needs_gemini_profile(row, **kwargs))
