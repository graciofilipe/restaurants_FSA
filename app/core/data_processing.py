import datetime
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
    for k in sorted(LONDON_OUTCODE_CENTROIDS.keys(), key=len, reverse=True):
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

def calculate_restaurant_priority(
    df: pd.DataFrame,
    anchor_lat: Optional[float] = None,
    anchor_lon: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    today_date: Optional[datetime.date] = None
) -> pd.DataFrame:
    """
    Computes distance, proximity score, staleness score, Google Maps prior, and composite priority score.
    Returns the DataFrame augmented with 'distance_km' and 'priority_score'.
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

    distances = []
    prox_scores = []
    stale_scores = []
    prior_scores = []
    scope_scores = []
    priority_scores = []

    for _, row in res_df.iterrows():
        # 1. Proximity & Distance
        lat = row.get('latitude')
        lon = row.get('longitude')
        has_exact = False
        if pd.notna(lat) and pd.notna(lon):
            try:
                lat_f, lon_f = float(lat), float(lon)
                if 45.0 <= lat_f <= 60.0 and -10.0 <= lon_f <= 5.0:
                    dist = haversine_distance_km(lat_f, lon_f, anchor_lat, anchor_lon)
                    has_exact = True
            except (ValueError, TypeError):
                has_exact = False

        if not has_exact:
            pc = first_present(row, 'postcode', 'PostCode') or ""
            centroid = lookup_outcode_coordinates(str(pc))
            # No coordinates and no placeable postcode. NaN, not 0.0: the grid
            # shows a blank distance and the sorts already put NaN last.
            dist = (haversine_distance_km(centroid[0], centroid[1], anchor_lat, anchor_lon)
                    if centroid else float('nan'))

        distances.append(dist)
        s_prox = (UNKNOWN_LOCATION_PROXIMITY_SCORE if math.isnan(dist)
                  else round(100.0 * math.exp(-0.20 * dist), 1))
        prox_scores.append(s_prox)

        # 2. Staleness & Re-scoring (100 for unscored, 80 for >=60d, 60 for >=30d, 40 for >=14d, 15 for recent)
        pred_val = row.get('predicted_user_rating')
        # "Has this been profiled?" is the timestamp the V2 merge writes.
        # `gemini_insights or gemini_insights_structured` was two wrong answers
        # at once (D-18): the `or` returned NaN for the 10,152 rows with no V1
        # text, so 1,020 profiled *and* predicted rows read as never scored and
        # kept the maximum staleness; and the 1,116 rows that hold only V1 text
        # read as profiled when no V2 profile exists. The stamp is exactly
        # co-extensive with `gemini_insights_structured` (2,767 rows, 0
        # disagreement either way) and survives the Phase 10 column drop.
        gemini_val = row.get('gemini_profiled_at')
        if pd.isna(pred_val) or pd.isna(gemini_val) or pred_val is None or gemini_val is None:
            s_stale = 100.0
        else:
            score_ts = row.get('predicted_at') if pd.notna(row.get('predicted_at')) else row.get('first_seen')
            days_ago = 45 # Default medium staleness
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

            if days_ago >= 60:
                s_stale = 80.0
            elif days_ago >= 30:
                s_stale = 60.0
            elif days_ago >= 14:
                s_stale = 40.0
            else:
                s_stale = 15.0
        stale_scores.append(s_stale)

        # 3. Google Maps Quality Prior (FSA excluded)
        mr = row.get('maps_rating')
        if pd.notna(mr):
            try:
                base_mr = max(0.0, (float(mr) - 3.0) * 50.0)
                mrev = row.get('maps_reviews')
                rev_num = float(mrev) if pd.notna(mrev) else 0.0
                boost = min(15.0, math.log10(max(1.0, rev_num + 1.0)) * 5.0)
                s_prior = round(min(100.0, base_mr + boost), 1)
            except Exception:
                s_prior = 50.0
        else:
            s_prior = 50.0
        prior_scores.append(s_prior)

        # 4. Scope Confidence
        in_sc = row.get('in_scope')
        is_out_of_scope = False
        if in_sc is True or in_sc == 1 or str(in_sc).lower() in ("true", "1"):
            s_scope = 100.0
        elif in_sc is False or in_sc == 0 or str(in_sc).lower() in ("false", "0"):
            s_scope = 0.0
            is_out_of_scope = True
        else:
            s_scope = 50.0
        scope_scores.append(s_scope)

        # Composite Priority Score
        if is_out_of_scope:
            p = 0.0
        else:
            p = round((w_prox * s_prox) + (w_stale * s_stale) + (w_prior * s_prior) + (w_scope * s_scope), 1)
            # If restaurant already has a human user_rating, lower priority for ML scoring
            user_rt = row.get('user_rating')
            if pd.notna(user_rt) and str(user_rt).strip() != "":
                p = round(p * 0.1, 1)

        priority_scores.append(p)

    res_df['distance_km'] = distances
    res_df['priority_score'] = priority_scores
    res_df['proximity_score'] = prox_scores
    res_df['staleness_score'] = stale_scores
    res_df['maps_prior_score'] = prior_scores

    return res_df
