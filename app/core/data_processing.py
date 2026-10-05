import datetime
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.core.pillar_schema import ALL_COLUMNS
from app.services.api_client import fetch_api_data
from app.services.bq_utils import ORIGINAL_COLUMNS_TO_KEEP

logger = logging.getLogger(__name__)

# The FSA search returns at most a few thousand establishments within the
# configured radius, so this is a runaway guard, not an expected limit.
DEFAULT_MAX_PAGES = 50

def parse_coordinates(coordinate_pairs_str: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Parses newline-separated coordinate pairs (lon, lat)."""
    valid_coords, errors = [], []
    for i, line in enumerate(coordinate_pairs_str.strip().split('\n')):
        line = line.strip()
        if not line:
            continue
        try:
            lon, lat = line.split(',')
            valid_coords.append((float(lon.strip()), float(lat.strip())))
        except ValueError:
            errors.append(f"Error parsing coordinate line {i+1}: '{line}'.")
    return valid_coords, errors

def fetch_data_for_all_coordinates(
    valid_coords: List[Tuple[float, float]], max_results: int, max_pages: int = DEFAULT_MAX_PAGES
) -> List[Dict[str, Any]]:
    """Fetches and aggregates API data for coordinates.

    `max_pages` bounds each coordinate independently. Without it, an API that
    keeps returning full pages -- or ignores the page parameter -- pages until
    the job is killed, sleeping a second and growing the result list each time.
    """
    all_establishments = []
    for lon, lat in valid_coords:
        for page in range(1, max_pages + 1):
            resp = fetch_api_data(lon, lat, max_results, page)
            time.sleep(1)
            if not resp:
                break
            ests = resp.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', []) or []
            all_establishments.extend(ests)
            if len(ests) < max_results:
                break
        else:
            logger.warning(
                f"Hit the {max_pages}-page limit for ({lon}, {lat}) without reaching a short page. "
                f"Results may be truncated, or the API may be ignoring the page parameter."
            )
    return all_establishments

def normalize_fhrsid(value: Any) -> str:
    """FHRSIDs reach us as ints from the API and strings from BigQuery."""
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return str(value).strip().lower()

def extract_fsa_coordinates(est: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Flattens the FSA API's nested `Geocode` into (latitude, longitude).

    Every establishment the API returns carries one, and until now the ingest
    dropped it -- `ORIGINAL_COLUMNS_TO_KEEP` is a flat key copy, so a nested
    object cannot survive it. We then paid Google Places to tell us where the
    restaurant was. The API sends the numbers quoted; `latitude`/`longitude`
    are FLOAT64, so anything unparseable becomes None rather than 0.0.
    """
    geocode = est.get('Geocode') or est.get('geocode') or {}
    if not isinstance(geocode, dict):
        return None, None

    def _as_float(*keys: str) -> Optional[float]:
        for key in keys:
            value = geocode.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    return _as_float('Latitude', 'latitude'), _as_float('Longitude', 'longitude')


def process_and_update_master_data(
    master_data: Iterable[Any], api_data: Dict[str, Any], today_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Processes API data to identify newly added establishments.

    `master_data` is whatever we already hold: either full rows or, from the
    weekly cron, just the FHRSIDs.
    """
    today_date = today_date or datetime.datetime.now().strftime("%Y-%m-%d")
    raw_ests = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    messages = []
    if raw_ests is None:
        api_ests = []
        messages.append("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not raw_ests:
        api_ests = []
        messages.append("API response contained no establishments in 'EstablishmentDetail'.")
    else:
        api_ests = raw_ests

    existing_ids = set()
    for est in master_data:
        # The cron passes bare FHRSIDs (a `SELECT fhrsid`); other callers pass whole rows.
        fid = (est.get('FHRSID') or est.get('fhrsid')) if isinstance(est, dict) else est
        if fid is not None:
            existing_ids.add(normalize_fhrsid(fid))

    new_records = []
    processed_ids = set()
    for est in api_ests:
        if isinstance(est, dict) and est.get('FHRSID') is not None:
            cid = normalize_fhrsid(est['FHRSID'])
            est['FHRSID'] = cid
            if cid not in existing_ids and cid not in processed_ids:
                est['first_seen'] = today_date
                est['latitude'], est['longitude'] = extract_fsa_coordinates(est)
                new_records.append({k: est.get(k) for k in ORIGINAL_COLUMNS_TO_KEEP})
                processed_ids.add(cid)

    count = len(new_records)
    if count > 0:
        summary_msg = f"Processed API response. Identified {count} unique new restaurant records to be added."
    elif messages:
        summary_msg = " ".join(messages)
    else:
        summary_msg = "Processed API response. No new restaurant records identified (or all were duplicates within the batch or already in BigQuery)."
    return new_records, summary_msg

def parse_bq_path(bq_path: str) -> Tuple[str, str, str]:
    """Parses 'project.dataset.table' format."""
    parts = bq_path.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid BigQuery Path: '{bq_path}'. Expected format: 'project.dataset.table'")
    return parts[0], parts[1], parts[2]

def enhance_dataframe_with_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantees the frame carries every pillar column the UI displays.

    The profile lands in typed columns at merge time (Phase 6), so the reader's
    whole job is now making sure they are present. It used to run `json.loads`
    on `gemini_insights_structured` once per row per Streamlit rerun and
    re-derive the pillars into a flat naming convention that existed only in
    this DataFrame -- a fourth spelling of the same six pillars, and the reason
    a renamed key could go unnoticed.

    A column that is absent is filled with NA, never with 0: no profile and a
    score of zero are different facts, and conflating them is D2.
    """
    if df is None or df.empty:
        return df
    missing = [column for column in ALL_COLUMNS if column not in df.columns]
    if not missing:
        return df
    return df.assign(**{column: pd.NA for column in missing})

import math
import os
import re

# Load comprehensive UK / Greater London outcode centroids
_OUTCODES_JSON_PATH = os.path.join(os.path.dirname(__file__), "london_outcodes.json")
LONDON_OUTCODE_CENTROIDS: Dict[str, Tuple[float, float]] = {}

if os.path.exists(_OUTCODES_JSON_PATH):
    try:
        with open(_OUTCODES_JSON_PATH, "r", encoding="utf-8") as _f:
            _raw_coords = json.load(_f)
            LONDON_OUTCODE_CENTROIDS = {k.upper(): (float(v[0]), float(v[1])) for k, v in _raw_coords.items()}
    except Exception as _e:
        pass

# Core fallbacks if json missing
if not LONDON_OUTCODE_CENTROIDS:
    LONDON_OUTCODE_CENTROIDS = {
        "SW16": (51.4277, -0.1294),
        "SW2": (51.4500, -0.1200),
        "SW4": (51.4600, -0.1400),
        "SW8": (51.4750, -0.1300),
        "SW9": (51.4650, -0.1150),
        "SW11": (51.4650, -0.1650),
        "SW12": (51.4450, -0.1500),
        "SW17": (51.4300, -0.1650),
        "SW19": (51.4200, -0.2050),
        "SE1": (51.4990, -0.0900),
        "SE5": (51.4700, -0.0900),
        "SE11": (51.4880, -0.1100),
        "SE15": (51.4700, -0.0650),
        "SE24": (51.4550, -0.1000),
        "SE27": (51.4350, -0.1050),
        "EC1": (51.5230, -0.0980),
        "EC2": (51.5180, -0.0850),
        "WC1": (51.5220, -0.1220),
        "WC2": (51.5120, -0.1240),
        "W1": (51.5150, -0.1420),
        "W2": (51.5160, -0.1780),
        "N1": (51.5380, -0.1020),
        "E1": (51.5150, -0.0600),
        "E2": (51.5300, -0.0600),
        "E8": (51.5450, -0.0750),
    }

# What `extract_outcode` says when there is no postcode to read. It used to say
# "SW16" -- the anchor's own outcode, 0 km from home and the maximum proximity
# score (D4). In the live table that path is latent rather than observed: a
# BigQuery NULL arrives in the frame as NaN, NaN is truthy, so the old
# `postcode or PostCode` read handed 'nan' to the lookup and it resolved to the
# central-London fallback instead -- 9.59 km, score 14.7. Both numbers are
# inventions. 126 rows carry no postcode and 2 carry junk ('WATERLOOVI', 'NE').
UNKNOWN_OUTCODE = "UNKNOWN"

# The prefix search order for `lookup_outcode_coordinates`, sorted once at
# import rather than once per lookup. 310 keys re-sorted for every unplaceable
# row is the hot loop D8 names; the sort is stable and the dictionary is never
# mutated after import, so hoisting it cannot change which prefix wins.
_OUTCODE_PREFIXES_LONGEST_FIRST: Tuple[str, ...] = tuple(
    sorted(LONDON_OUTCODE_CENTROIDS.keys(), key=len, reverse=True))

# What an unplaceable row scores for proximity instead. The decay is
# `100 * exp(-0.20 * km)`, so this is what a restaurant ~11.5 km from the anchor
# gets: behind anything we can actually place nearby, but still in the queue,
# because "we do not know where this is" is not a reason never to look at it.
UNKNOWN_LOCATION_PROXIMITY_SCORE = 10.0


def first_present(row: Any, *keys: str) -> Any:
    """First value in `row` that is neither absent nor NaN.

    `row.get('a') or row.get('b')` cannot be used on a DataFrame row: pandas
    fills a missing string with `float('nan')`, and NaN is *truthy*, so the
    chain stops on the missing value and returns it. That is D-18 -- the
    staleness rule read `gemini_insights or gemini_insights_structured` and so
    scored 10,152 rows as never profiled, 1,020 of which had a profile.
    """
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if not isinstance(value, str) and pd.isna(value):
            continue
        return value
    return None


def extract_outcode(postcode_str: str) -> str:
    """Extracts the UK outcode from a postcode string (e.g. 'SW4 7UL' -> 'SW4', 'SW196NW' -> 'SW19').

    Returns `UNKNOWN_OUTCODE` when there is nothing to extract.
    """
    if postcode_str is None or (not isinstance(postcode_str, str) and pd.isna(postcode_str)):
        return UNKNOWN_OUTCODE
    s = str(postcode_str).strip().upper()
    if not s:
        return UNKNOWN_OUTCODE
    if ' ' in s:
        return s.split(' ')[0].strip() or UNKNOWN_OUTCODE
    clean = re.sub(r'[^A-Z0-9]', '', s)
    if len(clean) >= 5 and re.match(r'^[A-Z0-9]+[0-9][A-Z]{2}$', clean):
        return clean[:-3]
    return clean or UNKNOWN_OUTCODE

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance between two points in kilometers."""
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (ValueError, TypeError):
        return 5.0
    r = 6371.0 # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)

def lookup_outcode_coordinates(outcode: str) -> Optional[Tuple[float, float]]:
    """Centroid (lat, lon) for a UK outcode or postcode, or None if we cannot place it.

    The None is the point. Scoring a restaurant needs to know the difference
    between "5 km away" and "no idea", and the old central-London fallback
    (Trafalgar Square) answered the second question with a number. 106 rows
    have neither coordinates nor a postcode we can place: 104 with no postcode
    at all and 2 carrying junk -- 'WATERLOOVI' and 'NE'.
    """
    if outcode is None or (not isinstance(outcode, str) and pd.isna(outcode)):
        return None

    clean_oc = extract_outcode(outcode)
    if clean_oc == UNKNOWN_OUTCODE:
        return None
    if clean_oc in LONDON_OUTCODE_CENTROIDS:
        return tuple(LONDON_OUTCODE_CENTROIDS[clean_oc])

    # Try longest prefix match (e.g. EC2A -> EC2, SW1A -> SW1)
    for k in _OUTCODE_PREFIXES_LONGEST_FIRST:
        if clean_oc.startswith(k):
            return tuple(LONDON_OUTCODE_CENTROIDS[k])

    return None


def get_outcode_coordinates(outcode: str) -> Tuple[float, float]:
    """Centroid (lat, lon) for a UK outcode, defaulting to SW16.

    This is the *anchor* lookup: the UI's "Anchor Postcode" box, where an empty
    or half-typed value has to resolve to something and the documented default
    is home. Row locations go through `lookup_outcode_coordinates`, which is
    allowed to say it does not know.
    """
    return lookup_outcode_coordinates(outcode) or LONDON_OUTCODE_CENTROIDS.get(
        "SW16", (51.4212, -0.1292))

def _column_or_missing(df: pd.DataFrame, name: str) -> pd.Series:
    """`df[name]`, or an all-missing column of the right length and index.

    The scalar loop used `row.get(name)`, which quietly yields None for a
    column the frame does not have. Callers routinely hand this function a
    frame missing `maps_rating` or `in_scope` entirely, so that tolerance is
    load-bearing rather than incidental.
    """
    if name in df.columns:
        return df[name]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def _absent_mask(series: pd.Series) -> pd.Series:
    """Where `first_present` would skip a value: None, or non-string NaN.

    A string is never absent, even an empty one -- that is `first_present`'s
    rule, and the reason it exists (D-18).
    """
    is_str = series.map(lambda v: isinstance(v, str)).astype(bool)
    return (~is_str) & series.isna()


def _first_present_column(df: pd.DataFrame, *names: str) -> pd.Series:
    """`first_present` over whole columns instead of a row at a time.

    Still per-row in effect -- a frame carrying both `postcode` and `PostCode`
    can take one from each row -- but expressed as two masked assignments
    rather than 11,268 dictionary walks.
    """
    present = [n for n in names if n in df.columns]
    if not present:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    out = df[present[0]].astype(object).copy()
    for name in present[1:]:
        out = out.where(~_absent_mask(out), df[name])
    return out


def _per_distinct_value(series: pd.Series, fn, na_result):
    """Call `fn` once per distinct value rather than once per row.

    The scalar helpers this wraps -- outcode resolution, timestamp coercion,
    the `in_scope` truthiness ladder -- are the parts of the scoring loop whose
    rules live in awkward `try`/`except` and `isinstance` chains. Rewriting
    them as array expressions would mean re-deriving those rules, and a
    re-derivation can be subtly wrong in a way nothing here would notice.
    Mapping over `factorize`'s uniques keeps the original scalar function as
    the definition while still collapsing the work: a production frame holds
    far fewer distinct postcodes than rows, and a batch of predictions shares
    one `predicted_at` to the microsecond.

    `factorize` codes missing values as -1, which indexes the appended
    `na_result` -- so `fn` is never called on a NaN it was never called on
    before.
    """
    codes, uniques = pd.factorize(series)
    results = [fn(value) for value in uniques]
    results.append(na_result)
    return [results[code] for code in codes]


def _round_like_python(values, ndigits: int):
    """Element-wise `round()`, because `np.round` is not the same function.

    Python's `round` converts the float exactly to decimal and rounds half to
    even. `np.round` multiplies by a power of ten, applies `rint`, and divides
    back, and the scaled value is not always exactly representable -- so the
    two disagree on values that land on a boundary. Measured on the live
    table, `np.round` moved 703 of 11,268 composite scores by 0.1: small, but
    the priority queue is sorted on exactly these numbers and D8 is supposed
    to change none of them.

    A Python-level loop over the array is still one pass instead of the
    per-row object churn it replaced -- tens of milliseconds against the
    0.8 seconds the loop cost.
    """
    flat = np.asarray(values, dtype=float)
    rounded = np.array([round(v, ndigits) for v in flat.ravel().tolist()], dtype=float)
    return rounded.reshape(flat.shape)


def _haversine_km_array(lat, lon, anchor_lat: float, anchor_lon: float):
    """`haversine_distance_km` over arrays, to the same two decimal places.

    NaN in, NaN out: an unplaceable row has no distance, and saying so is the
    point of the component.
    """
    r = 6371.0  # Earth's radius in km
    phi1 = np.radians(lat)
    phi2 = np.radians(anchor_lat)
    delta_phi = np.radians(anchor_lat - lat)
    delta_lambda = np.radians(anchor_lon - lon)
    a = (np.sin(delta_phi / 2.0) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _round_like_python(r * c, 2)


def _staleness_days(score_ts: Any, curr_date: datetime.date) -> int:
    """How many days ago a row was scored; 45 when the timestamp is unreadable.

    Lifted unchanged out of the scoring loop. `predicted_at` reaches here as a
    tz-aware `Timestamp` from BigQuery, a `date`, an ISO string, or nothing at
    all, and the 45-day default is what an unparseable value falls back to.
    """
    days_ago = 45  # Default medium staleness
    if score_ts and pd.notna(score_ts):
        try:
            if isinstance(score_ts, str):
                score_date = datetime.datetime.strptime(score_ts[:10], "%Y-%m-%d").date()
            elif isinstance(score_ts, (datetime.date, datetime.datetime, pd.Timestamp)):
                score_date = score_ts.date() if hasattr(score_ts, 'date') else score_ts
            else:
                score_date = curr_date
            days_ago = max(0, (curr_date - score_date).days)
        except Exception:
            days_ago = 45
    return days_ago


def _scope_score(in_scope: Any) -> float:
    """100 for in scope, 0 for out, 50 for not yet triaged.

    Lifted unchanged. The ladder is wider than `is True` because `in_scope`
    arrives as a BigQuery BOOL, as 1/0 through a DataFrame upcast, and as a
    string from the triage widgets. A 0.0 here means out of scope and nothing
    else, which is what lets the caller read the verdict back off the score.
    """
    if in_scope is True or in_scope == 1 or str(in_scope).lower() in ("true", "1"):
        return 100.0
    if in_scope is False or in_scope == 0 or str(in_scope).lower() in ("false", "0"):
        return 0.0
    return 50.0


def compute_active_learning_voi_scores(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute decision-theoretic Active Learning Value-of-Information scores (0-100):
    `(desk_voi_score, visit_voi_score)`.

    1. `desk_voi_score` ("Value of Desk Rating" — Pure Epistemic Disagreement x Leverage):
       - Eligible: Unrated Stage-2 sit-down `RESTAURANT_DINING` candidates.
       - Driven by Query-by-Committee disagreement `|tree_pred - lin_pred|` and
         prior residual `|predicted_user_rating - prior(match_score)|`, amplified
         by borough label scarcity and Pillar 7 feature sparsity.
    2. `visit_voi_score` ("Value of Visiting" — 100% Statistical & Culinary, Zero Distance Decay):
       - Sub-Pathway A (Unrated Exploratory Visits, `predicted_user_rating >= 5.5`):
         Driven by 90th-percentile Bayesian Upper Confidence Bound (`UCB_90`) above
         the dining threshold (`5.0`), epistemic uncertainty, and visited-borough scarcity.
       - Sub-Pathway B (High-Rated Desk Confirmations, `rating_source != 'visited'` and `user_rating >= 6`):
         Driven by the ground-truth promotion value (`1x/2x -> 2x Tree / 4x Linear` weight)
         combining `(user_rating - 5.0)`, model residual `|user_rating - predicted_user_rating|`,
         committee uncertainty, and visited-borough scarcity.
    """
    if df is None or df.empty:
        empty = pd.Series([], dtype=float)
        return empty, empty.copy()

    # Stage-2 Sit-Down Plausibility Gate: must be explicitly profiled as sit-down RESTAURANT_DINING
    sit_down = _column_or_missing(df, 'pillar_is_sit_down')
    is_true_sit_down = sit_down.map(
        lambda v: v is True or str(v).lower() in ('true', '1')
    ).astype(bool)
    pred_sit_down = pd.to_numeric(_column_or_missing(df, 'predicted_is_sit_down'), errors='coerce')
    is_pred_non_sit_down = (pred_sit_down.notna() & (pred_sit_down < 0.5)).astype(bool)
    est_type = _column_or_missing(df, 'pillar_establishment_type')
    is_restaurant_dining = est_type.map(
        lambda v: isinstance(v, str) and v.strip() == 'RESTAURANT_DINING'
    ).astype(bool)
    in_scope_col = _column_or_missing(df, 'in_scope')
    is_out_of_scope = in_scope_col.map(
        lambda v: v is False or str(v).lower() in ('false', '0')
    ).astype(bool)

    tree_num = pd.to_numeric(_column_or_missing(df, 'tree_pred'), errors='coerce')
    lin_num = pd.to_numeric(_column_or_missing(df, 'lin_pred'), errors='coerce')
    has_both_preds = (tree_num.notna() & lin_num.notna()).to_numpy()
    # Stage-1 gated rows in BigQuery have both tree_pred and lin_pred set to _stage1_capped_score (<= 2.0)
    is_stage1_capped = (
        has_both_preds
        & (tree_num.fillna(99.0).to_numpy(dtype=float) <= 2.0)
        & (lin_num.fillna(99.0).to_numpy(dtype=float) <= 2.0)
        & (tree_num.fillna(-1.0).to_numpy(dtype=float) == lin_num.fillna(-2.0).to_numpy(dtype=float))
    )

    is_stage2 = (
        is_true_sit_down.to_numpy()
        & is_restaurant_dining.to_numpy()
        & ~is_pred_non_sit_down.to_numpy()
        & ~is_out_of_scope.to_numpy()
        & ~is_stage1_capped
    )

    # Rating & source status
    user_rating_num = pd.to_numeric(_column_or_missing(df, 'user_rating'), errors='coerce')
    is_rated = user_rating_num.notna().to_numpy()
    is_unrated = ~is_rated
    rating_source_str = (
        _column_or_missing(df, 'rating_source')
        .fillna('')
        .astype(str)
        .str.strip()
        .str.lower()
    )
    is_visited = is_rated & (rating_source_str == 'visited').to_numpy()
    is_desk_or_null_rated = is_rated & ~is_visited

    # Committee Epistemic Uncertainty (sigma_QBC)
    match_num = pd.to_numeric(_column_or_missing(df, 'match_score'), errors='coerce').fillna(50.0).clip(0.0, 100.0)
    prior_pred = (0.643 + 0.072 * match_num).clip(1.0, 10.0)
    hybrid_pred = pd.to_numeric(_column_or_missing(df, 'predicted_user_rating'), errors='coerce').fillna(prior_pred)

    comm_num = pd.to_numeric(_column_or_missing(df, 'pillar_community_score'), errors='coerce').fillna(5.0)
    rev_num = pd.to_numeric(_column_or_missing(df, 'maps_reviews'), errors='coerce').fillna(0.0).clip(lower=0.0)
    log_rev_scale = (np.log10(rev_num + 1.0) * 2.5).clip(1.0, 10.0)
    enclave_vs_hype_gap = (comm_num - log_rev_scale).abs() * (40.0 / 9.0)
    fallback_delta_models = 2.0 * (enclave_vs_hype_gap.to_numpy(dtype=float) / 100.0)

    tree_minus_lin = (tree_num.fillna(0.0) - lin_num.fillna(0.0)).abs().to_numpy(dtype=float)
    delta_models = np.where(has_both_preds, tree_minus_lin, fallback_delta_models)
    delta_prior = (hybrid_pred - prior_pred).abs().to_numpy(dtype=float)
    sigma_qbc = np.sqrt(0.60 * np.square(delta_models) + 0.40 * np.square(delta_prior))
    sigma_std = np.sqrt(0.25 * np.square(delta_models) + 0.35 * np.square(delta_prior))

    # Borough scarcity counts
    borough_col = _first_present_column(df, 'localauthorityname', 'LocalAuthorityName')
    borough_key = borough_col.fillna('').astype(str).str.strip().str.lower()
    has_borough = (borough_key != '').to_numpy()

    stg2_labeled_series = pd.Series((is_stage2 & is_rated).astype(float), index=df.index)
    visited_series = pd.Series(is_visited.astype(float), index=df.index)
    n_stg2_borough = np.where(
        has_borough,
        stg2_labeled_series.groupby(borough_key).transform('sum').to_numpy(dtype=float),
        0.0,
    )
    n_visited_borough = np.where(
        has_borough,
        visited_series.groupby(borough_key).transform('sum').to_numpy(dtype=float),
        0.0,
    )

    # Pillar 4 hyper-local specificity indicator (plus legacy fallback)
    geo_specificity = (
        _first_present_column(df, 'pillar_geo_specificity', 'pillar_dish_specificity')
        .fillna('')
        .astype(str)
        .str.strip()
        .str.upper()
    )
    i_sparse = (geo_specificity == 'HYPER_LOCAL_CITY').to_numpy(dtype=float)

    # Channel 1: desk_voi_score (0-100)
    desk_leverage = 1.0 + (0.75 / np.sqrt(1.0 + n_stg2_borough)) + (0.25 * i_sparse)
    raw_desk_voi = sigma_qbc * desk_leverage
    scaled_desk_voi = np.clip((raw_desk_voi / 4.40) * 100.0, 0.0, 100.0)
    desk_eligible = is_stage2 & is_unrated
    desk_scores = np.where(desk_eligible, _round_like_python(scaled_desk_voi, 1), 0.0)

    # Channel 2: visit_voi_score (0-100, zero distance penalty)
    visit_leverage = 1.0 + (0.80 / np.sqrt(1.0 + n_visited_borough))
    hybrid_arr = hybrid_pred.to_numpy(dtype=float)

    # Sub-Pathway A: Unrated Exploratory Visits (hybrid >= 5.5)
    ucb_90 = hybrid_arr + 1.28 * sigma_std
    raw_visit_new = np.maximum(0.0, ucb_90 - 5.0) * (1.0 + 0.35 * sigma_std) * visit_leverage
    scaled_visit_new = np.clip((raw_visit_new / 7.50) * 100.0, 0.0, 100.0)
    unrated_visit_eligible = is_stage2 & is_unrated & (hybrid_arr >= 5.5)

    # Sub-Pathway B: High-Rated Desk Confirmations (user_rating >= 6.0, not yet visited)
    y_desk = np.nan_to_num(user_rating_num.to_numpy(dtype=float), nan=0.0)
    delta_anchor = np.abs(y_desk - hybrid_arr) + 0.50 * sigma_std
    raw_visit_confirm = np.maximum(0.0, y_desk - 5.0) * (1.0 + 0.60 * delta_anchor) * visit_leverage
    scaled_visit_confirm = np.clip((raw_visit_confirm / 16.50) * 100.0, 0.0, 100.0)
    confirm_visit_eligible = is_stage2 & is_desk_or_null_rated & (y_desk >= 6.0)

    visit_combined = np.select(
        [unrated_visit_eligible, confirm_visit_eligible],
        [scaled_visit_new, scaled_visit_confirm],
        default=0.0,
    )
    visit_scores = np.where(visit_combined > 0.0, _round_like_python(visit_combined, 1), 0.0)

    return (
        pd.Series(desk_scores, index=df.index, dtype=float),
        pd.Series(visit_scores, index=df.index, dtype=float),
    )


def compute_plausible_conflict_score(df: pd.DataFrame) -> pd.Series:
    """Backward-compatible wrapper returning `desk_voi_score` (0-100)."""
    desk_voi, _ = compute_active_learning_voi_scores(df)
    return desk_voi


def calculate_restaurant_priority(
    df: pd.DataFrame,
    anchor_lat: Optional[float] = None,
    anchor_lon: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    today_date: Optional[datetime.date] = None
) -> pd.DataFrame:
    """
    Computes distance, proximity score, staleness score, Google Maps prior, composite priority score,
    and Active Learning VoI scores ('desk_voi_score', 'visit_voi_score', 'conflict_score').

    Column-at-a-time since D8. The Streamlit ML Predictions tab re-scores the
    whole frame on every rerun -- every slider drag, every checkbox -- so this
    ran once per interaction, not once per load. The four components are array
    expressions; the awkward coercions stay as the scalar functions above and
    run once per distinct value via `_per_distinct_value`.
    """
    if df is None or df.empty:
        return df

    sw16_c = LONDON_OUTCODE_CENTROIDS.get("SW16", (51.4212, -0.1292))
    try:
        anchor_lat = float(anchor_lat) if anchor_lat is not None else sw16_c[0]
        anchor_lon = float(anchor_lon) if anchor_lon is not None else sw16_c[1]
    except (ValueError, TypeError):
        anchor_lat, anchor_lon = sw16_c

    weights = weights or {"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10}
    w_prox = weights.get("prox", 0.35)
    w_stale = weights.get("stale", 0.35)
    w_prior = weights.get("prior", 0.20)
    w_scope = weights.get("scope", 0.10)
    total_w = w_prox + w_stale + w_prior + w_scope
    if total_w > 0:
        w_prox, w_stale, w_prior, w_scope = w_prox / total_w, w_stale / total_w, w_prior / total_w, w_scope / total_w

    curr_date = today_date or datetime.date.today()
    res_df = df.copy()

    # 1. Proximity & Distance
    lat = pd.to_numeric(_column_or_missing(res_df, 'latitude'), errors='coerce')
    lon = pd.to_numeric(_column_or_missing(res_df, 'longitude'), errors='coerce')
    # `errors='coerce'` stands in for the loop's `try: float(...)`: a string
    # that will not parse becomes NaN and fails the bounding box, exactly as
    # the raised ValueError used to leave `has_exact` False.
    has_exact = (lat.between(45.0, 60.0) & lon.between(-10.0, 5.0)).to_numpy()
    exact_dist = _haversine_km_array(lat.to_numpy(dtype=float),
                                     lon.to_numpy(dtype=float),
                                     anchor_lat, anchor_lon)

    # Rows with usable coordinates never consult their postcode, so masking
    # here is not just an optimisation -- it keeps the outcode lookup off the
    # 97% of the table that ships with coordinates from the FSA.
    postcodes = _first_present_column(res_df, 'postcode', 'PostCode')
    postcodes = postcodes.where(~has_exact, other=None)
    centroids = _per_distinct_value(
        postcodes,
        lambda pc: lookup_outcode_coordinates(str(pc) if pc else ""),
        None)
    cent_lat = np.array([c[0] if c else np.nan for c in centroids], dtype=float)
    cent_lon = np.array([c[1] if c else np.nan for c in centroids], dtype=float)
    # No coordinates and no placeable postcode. NaN, not 0.0: the grid shows a
    # blank distance and the sorts already put NaN last.
    fallback_dist = _haversine_km_array(cent_lat, cent_lon, anchor_lat, anchor_lon)

    distances = np.where(has_exact, exact_dist, fallback_dist)
    prox_scores = np.where(
        np.isnan(distances),
        UNKNOWN_LOCATION_PROXIMITY_SCORE,
        _round_like_python(100.0 * np.exp(-0.20 * np.nan_to_num(distances, nan=0.0)), 1))

    # 2. Staleness & Re-scoring (100 for unscored, 80 for >=60d, 60 for >=30d, 40 for >=14d, 15 for recent)
    predicted = _column_or_missing(res_df, 'predicted_user_rating')
    profiled = _column_or_missing(res_df, 'gemini_profiled_at')
    unscored = (predicted.isna() | profiled.isna()).to_numpy()

    predicted_at = _column_or_missing(res_df, 'predicted_at')
    score_ts = predicted_at.where(predicted_at.notna(),
                                  _column_or_missing(res_df, 'first_seen'))
    # Blanked for unscored rows so the coercion is never asked about a
    # timestamp the loop would not have looked at.
    score_ts = score_ts.astype(object).where(~unscored, other=None)
    days_ago = np.array(
        _per_distinct_value(score_ts, lambda ts: _staleness_days(ts, curr_date), 45),
        dtype=float)

    stale_scores = np.select(
        [unscored, days_ago >= 60, days_ago >= 30, days_ago >= 14],
        [100.0, 80.0, 60.0, 40.0],
        default=15.0)

    # 3. Google Maps Quality Prior (FSA excluded)
    maps_rating = _column_or_missing(res_df, 'maps_rating')
    maps_reviews = _column_or_missing(res_df, 'maps_reviews')
    rating_num = pd.to_numeric(maps_rating, errors='coerce')
    reviews_num = pd.to_numeric(maps_reviews, errors='coerce')
    unreadable = (rating_num.isna() | (maps_reviews.notna() & reviews_num.isna())).to_numpy()
    usable = maps_rating.notna().to_numpy() & ~unreadable
    base = np.maximum(0.0, (np.nan_to_num(rating_num.to_numpy(dtype=float)) - 3.0) * 50.0)
    reviews = np.nan_to_num(reviews_num.to_numpy(dtype=float), nan=0.0)
    boost = np.minimum(15.0, np.log10(np.maximum(1.0, reviews + 1.0)) * 5.0)
    prior_scores = np.where(usable, _round_like_python(np.minimum(100.0, base + boost), 1), 50.0)

    # 4. Scope Confidence
    scope_scores = np.array(
        _per_distinct_value(_column_or_missing(res_df, 'in_scope'), _scope_score, 50.0),
        dtype=float)
    is_out_of_scope = scope_scores == 0.0

    # Composite Priority Score
    priority_scores = _round_like_python(
        (w_prox * prox_scores) + (w_stale * stale_scores)
        + (w_prior * prior_scores) + (w_scope * scope_scores), 1)
    # If restaurant already has a human user_rating, lower priority for ML scoring
    user_rating = _column_or_missing(res_df, 'user_rating')
    already_rated = (user_rating.notna()
                     & (user_rating.astype(str).str.strip() != "")).to_numpy()
    priority_scores = np.where(already_rated,
                               _round_like_python(priority_scores * 0.1, 1),
                               priority_scores)
    priority_scores = np.where(is_out_of_scope, 0.0, priority_scores)

    desk_voi, visit_voi = compute_active_learning_voi_scores(res_df)

    res_df['distance_km'] = distances
    res_df['priority_score'] = priority_scores
    res_df['proximity_score'] = prox_scores
    res_df['staleness_score'] = stale_scores
    res_df['maps_prior_score'] = prior_scores
    res_df['desk_voi_score'] = desk_voi
    res_df['visit_voi_score'] = visit_voi
    res_df['conflict_score'] = desk_voi

    return res_df
