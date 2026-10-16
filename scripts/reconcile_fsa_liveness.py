"""Reconcile per-FHRSID liveness on `fsa_master` against the live UK FSA API.

Probes `GET https://api.ratings.food.gov.uk/Establishments/{fhrsid}` (v2):
- HTTP 200 -> `fsa_status = 'active'`, `delisted_at = NULL`, `last_seen = today`,
  and refreshes `businessname`, `addressline1..3`, `postcode`, `ratingvalue`,
  `newratingpending`.
- HTTP 404 -> `fsa_status = 'delisted'`, `delisted_at = COALESCE(delisted_at, today)`.
- 5xx / network timeout -> retried with exponential backoff; if still failing,
  the row is skipped (never marked delisted on transient errors).

Then runs the supersession & carry-over pass:
- Links each `delisted` row (`D`) to an `active` row (`A`) sharing the same
  normalised `LOWER(TRIM(businessname))` + `UPPER(TRIM(postcode))` within
  `ST_DISTANCE <= 100 m` (falling back to name + postcode when either row lacks
  coordinates). Two `active` rows are **never** linked.
- Carries `user_rating` and `rating_source` to the active row (when both old and
  new rows are rated, the **newest rating by `first_seen` wins**, while the old
  rating remains on the retired row).
- Carries `in_scope`, `user_rating_10pt_backup`, `user_rating_7pt_backup`,
  `gemini_insights_structured`, 6-pillar columns, and `maps_*` columns to the
  active row **only where the active row is NULL** (plus promoting `in_scope`
  to TRUE when an unrated active row inherits a rating from an `in_scope = TRUE`
  retired row).

Usage:
    .venv/bin/python -m scripts.reconcile_fsa_liveness --only-duplicates --dry-run
    .venv/bin/python -m scripts.reconcile_fsa_liveness --only-duplicates --execute
    .venv/bin/python -m scripts.reconcile_fsa_liveness --fhrsids 35993,1871661 --dry-run
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import math
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from google.api_core import exceptions as gcp_exceptions
from google.auth.exceptions import RefreshError
from google.cloud import bigquery
from google.oauth2 import credentials as oauth2_credentials
import requests

from app.core.data_processing import extract_fsa_coordinates
from app.core.pillar_schema import PILLAR_FIELDS

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
STAGING_TABLE_NAME = "fsa_liveness_check"
FSA_ESTABLISHMENT_URL = "https://api.ratings.food.gov.uk/Establishments/{fhrsid}"
FSA_HEADERS = {
    "x-api-version": "2",
    "Accept": "application/json",
    "User-Agent": "restaurants-fsa-liveness-reconciler/1.0",
}
MAX_SUPERSESSION_DISTANCE_M = 100.0

CARRY_OVER_NULL_ONLY_COLUMNS: Tuple[str, ...] = (
    "user_rating_10pt_backup",
    "user_rating_7pt_backup",
    "gemini_insights_structured",
    "gemini_profiled_at",
    "price_level",
    "maps_rating",
    "maps_reviews",
    "maps_url",
    "business_status",
    "website_url",
    "maps_types",
    "maps_found",
    "maps_lookup_at",
    "latitude",
    "longitude",
) + tuple(f.column for f in PILLAR_FIELDS)


def _gcloud_token_client(project_id: str) -> bigquery.Client:
    """Build a BigQuery client backed by `gcloud auth print-access-token`."""
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token", "filipegracio@google.com"],
        text=True,
    ).strip()
    creds = oauth2_credentials.Credentials(token=token)
    return bigquery.Client(project=project_id, credentials=creds)


def _run_query(client: bigquery.Client, project_id: str, sql: str):
    """Execute a BigQuery query, falling back to gcloud access token on ADC Forbidden/RefreshError."""
    try:
        return list(client.query(sql).result())
    except (RefreshError, gcp_exceptions.Forbidden):
        fallback_client = _gcloud_token_client(project_id)
        return list(fallback_client.query(sql).result())


def _sql_literal(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return "NULL"
        return str(val)
    escaped = str(val).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def haversine_meters(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> Optional[float]:
    """Compute great-circle distance in meters between two WGS84 coordinates."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    try:
        phi1, lam1 = math.radians(float(lat1)), math.radians(float(lon1))
        phi2, lam2 = math.radians(float(lat2)), math.radians(float(lon2))
    except (TypeError, ValueError):
        return None
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * 6371000.0 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def _clean_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def probe_establishment(
    fhrsid: str,
    session: Optional[requests.Session] = None,
    retries: int = 3,
    backoff: float = 0.5,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Probe a single FHRSID on the FSA API (`/Establishments/{id}`).

    Returns a dictionary with:
    - `status_code`: 200, 404, or None/int on error
    - `fsa_status`: `'active'` (200), `'delisted'` (404), or `None` (transient failure)
    - refreshed establishment metadata when `status_code == 200`
    """
    fid = str(fhrsid).strip()
    url = FSA_ESTABLISHMENT_URL.format(fhrsid=fid)
    http = session or requests
    last_error: Optional[str] = None
    last_status: Optional[int] = None

    for attempt in range(max(1, retries)):
        try:
            resp = http.get(url, headers=FSA_HEADERS, timeout=timeout)
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json() if resp.content else {}
                lat_val, lon_val = (
                    extract_fsa_coordinates(data)
                    if isinstance(data, dict)
                    else (None, None)
                )
                nrp_raw = data.get("NewRatingPending") if isinstance(data, dict) else None
                if isinstance(nrp_raw, str):
                    nrp_val: Optional[bool] = nrp_raw.strip().lower() == "true"
                elif isinstance(nrp_raw, bool):
                    nrp_val = nrp_raw
                else:
                    nrp_val = None
                rating_raw = data.get("RatingValue") if isinstance(data, dict) else None
                return {
                    "fhrsid": fid,
                    "status_code": 200,
                    "fsa_status": "active",
                    "businessname": _clean_str(data.get("BusinessName")) if isinstance(data, dict) else None,
                    "addressline1": _clean_str(data.get("AddressLine1")) if isinstance(data, dict) else None,
                    "addressline2": _clean_str(data.get("AddressLine2")) if isinstance(data, dict) else None,
                    "addressline3": _clean_str(data.get("AddressLine3")) if isinstance(data, dict) else None,
                    "postcode": _clean_str(data.get("PostCode")) if isinstance(data, dict) else None,
                    "ratingvalue": _clean_str(rating_raw),
                    "newratingpending": nrp_val,
                    "latitude": lat_val,
                    "longitude": lon_val,
                    "error": None,
                }
            if resp.status_code == 404:
                return {
                    "fhrsid": fid,
                    "status_code": 404,
                    "fsa_status": "delisted",
                    "businessname": None,
                    "addressline1": None,
                    "addressline2": None,
                    "addressline3": None,
                    "postcode": None,
                    "ratingvalue": None,
                    "newratingpending": None,
                    "latitude": None,
                    "longitude": None,
                    "error": None,
                }
            last_error = f"HTTP {resp.status_code}"
        except Exception as exc:
            last_error = str(exc)

        if attempt < retries - 1 and backoff > 0:
            time.sleep(backoff * (2 ** attempt))

    return {
        "fhrsid": fid,
        "status_code": last_status,
        "fsa_status": None,
        "businessname": None,
        "addressline1": None,
        "addressline2": None,
        "addressline3": None,
        "postcode": None,
        "ratingvalue": None,
        "newratingpending": None,
        "latitude": None,
        "longitude": None,
        "error": last_error or "Unknown probe error",
    }


def probe_establishments(
    fhrsids: Sequence[str],
    session: Optional[requests.Session] = None,
    retries: int = 4,
    backoff: float = 1.0,
    timeout: float = 10.0,
    max_workers: int = 1,
    sleep_seconds: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    """Probe a collection of FHRSIDs against the FSA API."""
    unique_ids = [str(fid).strip() for fid in dict.fromkeys(fhrsids) if str(fid).strip()]
    results: Dict[str, Dict[str, Any]] = {}
    if not unique_ids:
        return results

    if session is not None or max_workers <= 1:
        own_session = session is None
        sess = session or requests.Session()
        try:
            for idx, fid in enumerate(unique_ids):
                results[fid] = probe_establishment(
                    fid, session=sess, retries=retries, backoff=backoff, timeout=timeout
                )
                if sleep_seconds > 0 and idx < len(unique_ids) - 1:
                    time.sleep(sleep_seconds)
        finally:
            if own_session:
                sess.close()
        return results

    tls = threading.local()

    def _worker_probe(fid: str) -> Dict[str, Any]:
        sess = getattr(tls, "session", None)
        if sess is None:
            sess = requests.Session()
            tls.session = sess
        return probe_establishment(
            fid,
            session=sess,
            retries=retries,
            backoff=backoff,
            timeout=timeout,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_id = {
            pool.submit(_worker_probe, fid): fid
            for fid in unique_ids
        }
        for fut in as_completed(future_to_id):
            fid = future_to_id[fut]
            results[fid] = fut.result()
    return results


def _norm_name(val: Any) -> str:
    return str(val or "").strip().lower()


def _norm_postcode(val: Any) -> str:
    return str(val or "").strip().upper()


def _can_supersede(delisted_row: Dict[str, Any], active_row: Dict[str, Any]) -> Tuple[bool, float]:
    """Return `(matches, distance_m_or_0)` for a delisted row and an active candidate."""
    if delisted_row.get("fsa_status") != "delisted" or active_row.get("fsa_status") != "active":
        return False, float("inf")
    if str(delisted_row.get("fhrsid")) == str(active_row.get("fhrsid")):
        return False, float("inf")

    d_name = _norm_name(delisted_row.get("businessname"))
    a_name = _norm_name(active_row.get("businessname"))
    if not d_name or d_name != a_name:
        return False, float("inf")

    d_pc = _norm_postcode(delisted_row.get("postcode"))
    a_pc = _norm_postcode(active_row.get("postcode"))
    dist = haversine_meters(
        delisted_row.get("latitude"),
        delisted_row.get("longitude"),
        active_row.get("latitude"),
        active_row.get("longitude"),
    )

    if d_pc and a_pc:
        if d_pc != a_pc:
            return False, float("inf")
        if dist is not None and dist > MAX_SUPERSESSION_DISTANCE_M:
            return False, float("inf")
        return True, dist if dist is not None else 0.0

    # Both postcodes missing: only link when both have coordinates within 100 m
    if not d_pc and not a_pc and dist is not None and dist <= MAX_SUPERSESSION_DISTANCE_M:
        return True, dist

    return False, float("inf")


def _row_sort_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    first_seen = str(row.get("first_seen") or "")
    fid_str = str(row.get("fhrsid") or "")
    try:
        fid_int = int(fid_str)
    except ValueError:
        fid_int = -1
    return (first_seen, fid_int, fid_str)


def reconcile_in_memory(
    rows: Iterable[Dict[str, Any]],
    probe_results: Dict[str, Dict[str, Any]],
    today: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Apply probe liveness updates, supersession links, and carry-over in memory.

    Used by `--dry-run` reporting and unit tests.
    """
    today_str = today or datetime.date.today().isoformat()
    working: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        r = dict(raw)
        fid = str(r["fhrsid"])
        r["fhrsid"] = fid
        r.setdefault("fsa_status", "active")
        r.setdefault("last_seen", r.get("first_seen"))
        r.setdefault("delisted_at", None)
        r.setdefault("superseded_by", None)
        working[fid] = r

    # Step 1: Apply probe statuses
    for fid, r in working.items():
        probe = probe_results.get(fid)
        if not probe or probe.get("fsa_status") is None:
            continue
        if probe["fsa_status"] == "delisted":
            r["fsa_status"] = "delisted"
            if not r.get("delisted_at"):
                r["delisted_at"] = today_str
        elif probe["fsa_status"] == "active":
            r["fsa_status"] = "active"
            r["delisted_at"] = None
            r["last_seen"] = today_str
            for col in (
                "businessname",
                "addressline1",
                "addressline2",
                "addressline3",
                "postcode",
                "ratingvalue",
                "newratingpending",
            ):
                if probe.get(col) is not None:
                    r[col] = probe[col]

    # Step 2: Link each delisted row to best matching active row (never link two active rows)
    active_rows = [r for r in working.values() if r.get("fsa_status") == "active"]
    delisted_rows = [r for r in working.values() if r.get("fsa_status") == "delisted"]

    superseded_by_target: Dict[str, List[Dict[str, Any]]] = {}
    for d_row in delisted_rows:
        candidates: List[Tuple[float, Tuple[str, int, str], Dict[str, Any]]] = []
        for a_row in active_rows:
            ok, dist = _can_supersede(d_row, a_row)
            if ok:
                candidates.append((dist, _row_sort_key(a_row), a_row))
        if candidates:
            # Prefer closest distance, then newest first_seen / highest fhrsid
            candidates.sort(key=lambda item: (item[0], -item[1][1], item[1][0]))
            best_dist = candidates[0][0]
            best_matches = [c for c in candidates if abs(c[0] - best_dist) < 1e-6]
            best_matches.sort(key=lambda item: item[1], reverse=True)
            winner_active = best_matches[0][2]
            target_id = str(winner_active["fhrsid"])
            d_row["superseded_by"] = target_id
            superseded_by_target.setdefault(target_id, []).append(d_row)

    # Step 3: Carry over from delisted rows to their active replacement
    for target_id, d_list in superseded_by_target.items():
        a_row = working[target_id]
        d_sorted_newest = sorted(d_list, key=_row_sort_key, reverse=True)

        # 3a. user_rating & rating_source: newest rating by first_seen wins;
        # old value stays on the retired row.
        rated_d = [d for d in d_sorted_newest if d.get("user_rating") is not None]
        if rated_d:
            best_d = rated_d[0]
            if a_row.get("user_rating") is None:
                a_row["user_rating"] = best_d["user_rating"]
                if best_d.get("rating_source") is not None:
                    a_row["rating_source"] = best_d["rating_source"]
                if best_d.get("in_scope") is True:
                    a_row["in_scope"] = True
            else:
                # Both active and delisted are rated -> newest by first_seen wins
                a_first = str(a_row.get("first_seen") or "")
                d_first = str(best_d.get("first_seen") or "")
                if d_first > a_first:
                    a_row["user_rating"] = best_d["user_rating"]
                    if best_d.get("rating_source") is not None:
                        a_row["rating_source"] = best_d["rating_source"]

        # 3b. in_scope: target-NULL-only
        if a_row.get("in_scope") is None:
            for d in d_sorted_newest:
                if d.get("in_scope") is not None:
                    a_row["in_scope"] = d["in_scope"]
                    break

        # 3c. All other carried columns: target-NULL-only
        for col in CARRY_OVER_NULL_ONLY_COLUMNS:
            cur_val = a_row.get(col)
            is_empty = cur_val is None or (
                col == "gemini_insights_structured" and str(cur_val).strip() == ""
            )
            if is_empty:
                for d in d_sorted_newest:
                    d_val = d.get(col)
                    if d_val is not None and not (
                        col == "gemini_insights_structured" and str(d_val).strip() == ""
                    ):
                        a_row[col] = d_val
                        break

    # Step 4: For active rows that still lack coordinates, fill from probe if available
    for fid, r in working.items():
        if r.get("fsa_status") == "active":
            probe = probe_results.get(fid) or {}
            if r.get("latitude") is None and probe.get("latitude") is not None:
                r["latitude"] = probe["latitude"]
            if r.get("longitude") is None and probe.get("longitude") is not None:
                r["longitude"] = probe["longitude"]

    return list(working.values())


def build_candidates_query(
    bq_path: str,
    only_duplicates: bool = False,
    only_non_standard_ratings: bool = False,
    fhrsids: Optional[Sequence[str]] = None,
) -> str:
    """Build SQL to select rows to probe for liveness."""
    if fhrsids:
        quoted = ", ".join(_sql_literal(str(f).strip()) for f in fhrsids if str(f).strip())
        where_clause = f"WHERE fhrsid IN ({quoted})"
    elif only_duplicates:
        where_clause = f"""
        WHERE CONCAT(IFNULL(businessname, ''), '|', IFNULL(postcode, '')) IN (
          SELECT CONCAT(IFNULL(businessname, ''), '|', IFNULL(postcode, ''))
          FROM `{bq_path}`
          GROUP BY 1
          HAVING COUNT(*) > 1
        )
        """
    elif only_non_standard_ratings:
        where_clause = """
        WHERE fsa_status IS DISTINCT FROM 'delisted'
          AND (ratingvalue IS NULL OR ratingvalue NOT IN ('1', '2', '3', '4', '5'))
        """
    else:
        where_clause = ""

    return f"""
    SELECT *
    FROM `{bq_path}`
    {where_clause}
    ORDER BY businessname, postcode, first_seen, fhrsid
    """


def _probe_structs_sql(
    valid: Sequence[Dict[str, Any]],
    today_str: str,
) -> str:
    structs: List[str] = []
    for p in valid:
        structs.append(
            "STRUCT("
            f"{_sql_literal(str(p['fhrsid']))} AS fhrsid, "
            f"CAST({_sql_literal(p.get('status_code'))} AS INT64) AS status_code, "
            f"{_sql_literal(p.get('fsa_status'))} AS fsa_status, "
            f"CAST({_sql_literal(p.get('businessname'))} AS STRING) AS businessname, "
            f"CAST({_sql_literal(p.get('addressline1'))} AS STRING) AS addressline1, "
            f"CAST({_sql_literal(p.get('addressline2'))} AS STRING) AS addressline2, "
            f"CAST({_sql_literal(p.get('addressline3'))} AS STRING) AS addressline3, "
            f"CAST({_sql_literal(p.get('postcode'))} AS STRING) AS postcode, "
            f"CAST({_sql_literal(p.get('ratingvalue'))} AS STRING) AS ratingvalue, "
            f"CAST({_sql_literal(p.get('newratingpending'))} AS BOOL) AS newratingpending, "
            f"CAST({_sql_literal(p.get('latitude'))} AS FLOAT64) AS latitude, "
            f"CAST({_sql_literal(p.get('longitude'))} AS FLOAT64) AS longitude, "
            f"DATE('{today_str}') AS checked_at"
            ")"
        )
    return ",\n      ".join(structs)


def build_staging_table_sql(
    staging_path: str,
    probe_results: Sequence[Dict[str, Any]],
    today_str: str,
) -> str:
    """Build SQL to materialize `fsa_liveness_check` staging table in BigQuery."""
    valid = [p for p in probe_results if p.get("fsa_status") in ("active", "delisted")]
    if not valid:
        return f"""
        CREATE OR REPLACE TABLE `{staging_path}` AS
        SELECT
          CAST(NULL AS STRING) AS fhrsid,
          CAST(NULL AS INT64) AS status_code,
          CAST(NULL AS STRING) AS fsa_status,
          CAST(NULL AS STRING) AS businessname,
          CAST(NULL AS STRING) AS addressline1,
          CAST(NULL AS STRING) AS addressline2,
          CAST(NULL AS STRING) AS addressline3,
          CAST(NULL AS STRING) AS postcode,
          CAST(NULL AS STRING) AS ratingvalue,
          CAST(NULL AS BOOL) AS newratingpending,
          CAST(NULL AS FLOAT64) AS latitude,
          CAST(NULL AS FLOAT64) AS longitude,
          DATE('{today_str}') AS checked_at
        FROM (SELECT 1) WHERE FALSE
        """

    unnest_body = _probe_structs_sql(valid, today_str)
    return f"""
    CREATE OR REPLACE TABLE `{staging_path}` AS
    SELECT * FROM UNNEST([
      {unnest_body}
    ])
    """


def build_staging_insert_sql(
    staging_path: str,
    probe_results: Sequence[Dict[str, Any]],
    today_str: str,
) -> str:
    """Build `INSERT INTO` SQL for subsequent staging chunks when >500 rows are probed."""
    valid = [p for p in probe_results if p.get("fsa_status") in ("active", "delisted")]
    unnest_body = _probe_structs_sql(valid, today_str)
    return f"""
    INSERT INTO `{staging_path}`
    SELECT * FROM UNNEST([
      {unnest_body}
    ])
    """


def build_liveness_merge_sql(
    bq_path: str,
    staging_path: str,
    today_str: str,
) -> str:
    """Build `MERGE` SQL applying `fsa_liveness_check` results to `fsa_master`."""
    return f"""
    MERGE `{bq_path}` AS T
    USING `{staging_path}` AS S
    ON T.fhrsid = S.fhrsid
    WHEN MATCHED AND S.fsa_status = 'delisted' THEN
      UPDATE SET
        fsa_status = 'delisted',
        delisted_at = COALESCE(T.delisted_at, DATE('{today_str}'))
    WHEN MATCHED AND S.fsa_status = 'active' THEN
      UPDATE SET
        fsa_status = 'active',
        delisted_at = NULL,
        last_seen = DATE('{today_str}'),
        businessname = COALESCE(S.businessname, T.businessname),
        addressline1 = COALESCE(S.addressline1, T.addressline1),
        addressline2 = COALESCE(S.addressline2, T.addressline2),
        addressline3 = COALESCE(S.addressline3, T.addressline3),
        postcode = COALESCE(S.postcode, T.postcode),
        ratingvalue = COALESCE(S.ratingvalue, T.ratingvalue),
        newratingpending = COALESCE(S.newratingpending, T.newratingpending)
    """


def build_link_superseded_sql(
    bq_path: str,
    new_ids: Optional[Sequence[str]] = None,
) -> str:
    """Build SQL that sets `superseded_by` on `delisted` rows matching an `active` row.

    Match criteria:
    - `D.fsa_status = 'delisted'` AND `A.fsa_status = 'active'` (never links two active rows)
    - `LOWER(TRIM(D.businessname)) = LOWER(TRIM(A.businessname))`
    - Same `UPPER(TRIM(postcode))` (or both postcodes NULL with coordinates within 100 m)
    - `ST_DISTANCE <= 100 m` (or either row lacks `latitude`/`longitude` when postcode matches)
    """
    scope_clause = ""
    if new_ids:
        quoted = ", ".join(_sql_literal(str(fid).strip()) for fid in new_ids if str(fid).strip())
        if quoted:
            scope_clause = f"AND (A.fhrsid IN ({quoted}) OR D.fhrsid IN ({quoted}))"

    return f"""
    MERGE `{bq_path}` AS T
    USING (
      SELECT
        delisted_fhrsid,
        active_fhrsid
      FROM (
        SELECT
          D.fhrsid AS delisted_fhrsid,
          A.fhrsid AS active_fhrsid,
          ROW_NUMBER() OVER (
            PARTITION BY D.fhrsid
            ORDER BY
              COALESCE(
                ST_DISTANCE(
                  ST_GEOGPOINT(D.longitude, D.latitude),
                  ST_GEOGPOINT(A.longitude, A.latitude)
                ),
                0.0
              ) ASC,
              A.first_seen DESC,
              SAFE_CAST(A.fhrsid AS INT64) DESC,
              A.fhrsid DESC
          ) AS rn
        FROM `{bq_path}` AS D
        JOIN `{bq_path}` AS A
          ON D.fhrsid != A.fhrsid
         AND D.fsa_status = 'delisted'
         AND A.fsa_status = 'active'
         AND D.businessname IS NOT NULL
         AND A.businessname IS NOT NULL
         AND LOWER(TRIM(D.businessname)) = LOWER(TRIM(A.businessname))
         AND (
           (
             D.postcode IS NOT NULL AND A.postcode IS NOT NULL
             AND TRIM(D.postcode) != '' AND TRIM(A.postcode) != ''
             AND UPPER(TRIM(D.postcode)) = UPPER(TRIM(A.postcode))
           )
           OR (
             D.postcode IS NULL AND A.postcode IS NULL
             AND D.latitude IS NOT NULL AND D.longitude IS NOT NULL
             AND A.latitude IS NOT NULL AND A.longitude IS NOT NULL
           )
         )
         AND (
           D.latitude IS NULL OR D.longitude IS NULL
           OR A.latitude IS NULL OR A.longitude IS NULL
           OR ST_DISTANCE(
                ST_GEOGPOINT(D.longitude, D.latitude),
                ST_GEOGPOINT(A.longitude, A.latitude)
              ) <= {MAX_SUPERSESSION_DISTANCE_M}
         )
         {scope_clause}
      )
      WHERE rn = 1
    ) AS S
    ON T.fhrsid = S.delisted_fhrsid
    WHEN MATCHED AND T.superseded_by IS DISTINCT FROM S.active_fhrsid THEN
      UPDATE SET T.superseded_by = S.active_fhrsid
    """


def build_carry_over_sql(
    bq_path: str,
    new_ids: Optional[Sequence[str]] = None,
) -> str:
    """Build SQL that carries user ratings and metadata from superseded rows to active rows.

    - `user_rating` & `rating_source`: when only the delisted row is rated, copy to active;
      when both are rated, newest rating by `first_seen` wins (old value stays on delisted row).
    - `in_scope`: copied when active row is NULL, or set to TRUE when an unrated active row
      inherits a rating from an `in_scope = TRUE` delisted row.
    - All other carried columns (`user_rating_*_backup`, `gemini_insights_structured`,
      6-pillar columns, `maps_*`, `latitude`, `longitude`): copied only where active row is NULL.
    """
    scope_clause = ""
    if new_ids:
        quoted = ", ".join(_sql_literal(str(fid).strip()) for fid in new_ids if str(fid).strip())
        if quoted:
            scope_clause = f"AND (D.superseded_by IN ({quoted}) OR D.fhrsid IN ({quoted}))"

    agg_exprs: List[str] = []
    set_exprs: List[str] = [
        """user_rating = CASE
          WHEN T.user_rating IS NULL AND S.d_user_rating IS NOT NULL THEN S.d_user_rating
          WHEN T.user_rating IS NOT NULL AND S.d_user_rating IS NOT NULL
               AND S.d_rated_first_seen IS NOT NULL
               AND (T.first_seen IS NULL OR S.d_rated_first_seen > T.first_seen)
          THEN S.d_user_rating
          ELSE T.user_rating
        END""",
        """rating_source = CASE
          WHEN T.user_rating IS NULL AND S.d_user_rating IS NOT NULL
          THEN COALESCE(S.d_rating_source, T.rating_source)
          WHEN T.user_rating IS NOT NULL AND S.d_user_rating IS NOT NULL
               AND S.d_rated_first_seen IS NOT NULL
               AND (T.first_seen IS NULL OR S.d_rated_first_seen > T.first_seen)
          THEN COALESCE(S.d_rating_source, T.rating_source)
          ELSE COALESCE(T.rating_source, S.d_rating_source)
        END""",
        """in_scope = CASE
          WHEN T.user_rating IS NULL AND S.d_user_rating IS NOT NULL AND S.d_rated_in_scope IS TRUE
          THEN TRUE
          ELSE COALESCE(T.in_scope, S.d_in_scope)
        END""",
    ]

    for col in CARRY_OVER_NULL_ONLY_COLUMNS:
        if col == "gemini_insights_structured":
            agg_exprs.append(
                f"ARRAY_AGG(IF(TRIM(D.{col}) != '', D.{col}, NULL) IGNORE NULLS "
                f"ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC LIMIT 1)[SAFE_OFFSET(0)] AS d_{col}"
            )
            set_exprs.append(
                f"{col} = IF(T.{col} IS NOT NULL AND TRIM(T.{col}) != '', T.{col}, S.d_{col})"
            )
        else:
            agg_exprs.append(
                f"ARRAY_AGG(D.{col} IGNORE NULLS "
                f"ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC LIMIT 1)[SAFE_OFFSET(0)] AS d_{col}"
            )
            set_exprs.append(f"{col} = COALESCE(T.{col}, S.d_{col})")

    agg_sql = ",\n        ".join(agg_exprs)
    set_sql = ",\n        ".join(set_exprs)

    return f"""
    MERGE `{bq_path}` AS T
    USING (
      SELECT
        D.superseded_by AS active_fhrsid,
        ARRAY_AGG(
          D.user_rating IGNORE NULLS
          ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC
          LIMIT 1
        )[SAFE_OFFSET(0)] AS d_user_rating,
        ARRAY_AGG(
          IF(D.user_rating IS NOT NULL, D.rating_source, NULL) IGNORE NULLS
          ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC
          LIMIT 1
        )[SAFE_OFFSET(0)] AS d_rating_source,
        ARRAY_AGG(
          IF(D.user_rating IS NOT NULL, D.first_seen, NULL) IGNORE NULLS
          ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC
          LIMIT 1
        )[SAFE_OFFSET(0)] AS d_rated_first_seen,
        ARRAY_AGG(
          IF(D.user_rating IS NOT NULL, D.in_scope, NULL) IGNORE NULLS
          ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC
          LIMIT 1
        )[SAFE_OFFSET(0)] AS d_rated_in_scope,
        ARRAY_AGG(
          D.in_scope IGNORE NULLS
          ORDER BY D.first_seen DESC, SAFE_CAST(D.fhrsid AS INT64) DESC, D.fhrsid DESC
          LIMIT 1
        )[SAFE_OFFSET(0)] AS d_in_scope,
        {agg_sql}
      FROM `{bq_path}` AS D
      WHERE D.fsa_status = 'delisted'
        AND D.superseded_by IS NOT NULL
        {scope_clause}
      GROUP BY D.superseded_by
    ) AS S
    ON T.fhrsid = S.active_fhrsid AND T.fsa_status = 'active'
    WHEN MATCHED THEN
      UPDATE SET
        {set_sql}
    """


def print_reconciliation_report(
    before_rows: Sequence[Dict[str, Any]],
    after_rows: Sequence[Dict[str, Any]],
    mode_label: str,
) -> None:
    """Print before/after status counts, the rated duplicate groups, and the kept-live pairs."""
    before_active = sum(1 for r in before_rows if (r.get("fsa_status") or "active") == "active")
    before_delisted = sum(1 for r in before_rows if r.get("fsa_status") == "delisted")
    before_superseded = sum(1 for r in before_rows if r.get("superseded_by"))

    after_active = sum(1 for r in after_rows if (r.get("fsa_status") or "active") == "active")
    after_delisted = sum(1 for r in after_rows if r.get("fsa_status") == "delisted")
    after_superseded = sum(1 for r in after_rows if r.get("superseded_by"))

    print(f"\n=== FSA Liveness Reconciliation Report ({mode_label}) ===")
    print(
        f"Before (scoped rows={len(before_rows)}): "
        f"active={before_active}, delisted={before_delisted}, superseded={before_superseded}"
    )
    print(
        f"After  (scoped rows={len(after_rows)}): "
        f"active={after_active}, delisted={after_delisted}, superseded={after_superseded}"
    )

    orig_key_by_id: Dict[str, Tuple[str, str]] = {
        str(r.get("fhrsid")): (str(r.get("businessname") or ""), str(r.get("postcode") or ""))
        for r in before_rows
        if r.get("fhrsid") is not None
    }
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in after_rows:
        fid = str(r.get("fhrsid"))
        key = orig_key_by_id.get(
            fid, (str(r.get("businessname") or ""), str(r.get("postcode") or ""))
        )
        groups.setdefault(key, []).append(r)

    rated_groups = [
        (k, g)
        for k, g in sorted(groups.items())
        if len(g) > 1 and any(row.get("user_rating") is not None for row in g)
    ]
    print(f"\n--- Rated Duplicate Groups ({len(rated_groups)}) ---")
    for (name, postcode), g in rated_groups:
        row_Parts = []
        for r in sorted(g, key=_row_sort_key):
            row_Parts.append(
                f"{r['fhrsid']}[{r.get('fsa_status')} -> {r.get('superseded_by') or '-'} | "
                f"rating={r.get('user_rating')} ({r.get('rating_source')}) | in_scope={r.get('in_scope')}]"
            )
        print(f"  * {name} ({postcode}): " + "; ".join(row_Parts))

    both_live_groups = [
        (k, g)
        for k, g in sorted(groups.items())
        if len(g) > 1 and all(row.get("fsa_status") == "active" for row in g)
    ]
    print(f"\n--- Kept-Live Duplicate Pairs ({len(both_live_groups)}) ---")
    for (name, postcode), g in both_live_groups:
        row_Parts = [
            f"{r['fhrsid']}[name={r.get('businessname')}, status={r.get('fsa_status')}, superseded_by={r.get('superseded_by')}, addr={r.get('addressline1')}]"
            for r in sorted(g, key=_row_sort_key)
        ]
        print(f"  * {name} ({postcode}): " + "; ".join(row_Parts))


def reconcile_fsa_liveness(
    bq_path: str = DEFAULT_BQ_PATH,
    execute: bool = False,
    only_duplicates: bool = False,
    only_non_standard_ratings: bool = False,
    fhrsids: Optional[Sequence[str]] = None,
    max_workers: int = 1,
    sleep_seconds: float = 0.2,
) -> Dict[str, Any]:
    """Run liveness probing and (when `execute=True`) BigQuery MERGE + supersession."""
    parts = bq_path.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected project.dataset.table, got: {bq_path}")
    project_id, dataset_id, _ = parts
    staging_path = f"{project_id}.{dataset_id}.{STAGING_TABLE_NAME}"
    today_str = datetime.date.today().isoformat()

    try:
        client = bigquery.Client(project=project_id)
    except Exception:
        client = _gcloud_token_client(project_id)

    cand_sql = build_candidates_query(
        bq_path=bq_path,
        only_duplicates=only_duplicates,
        only_non_standard_ratings=only_non_standard_ratings,
        fhrsids=fhrsids,
    )
    before_rows = [dict(r) for r in _run_query(client, project_id, cand_sql)]
    target_ids = [str(r["fhrsid"]) for r in before_rows if r.get("fhrsid") is not None]
    print(f"Probing {len(target_ids)} FHRSID(s) on the FSA API (max_workers={max_workers})...")

    probe_map = probe_establishments(
        target_ids,
        max_workers=max_workers,
        sleep_seconds=sleep_seconds,
    )
    n_200 = sum(1 for p in probe_map.values() if p.get("status_code") == 200)
    n_404 = sum(1 for p in probe_map.values() if p.get("status_code") == 404)
    n_err = sum(1 for p in probe_map.values() if p.get("fsa_status") is None)
    print(f"Probe results: 200 (active)={n_200}, 404 (delisted)={n_404}, errors/skipped={n_err}")

    if not execute:
        after_rows = reconcile_in_memory(before_rows, probe_map, today=today_str)
        print_reconciliation_report(before_rows, after_rows, mode_label="DRY-RUN")
        return {
            "mode": "dry_run",
            "probed": len(target_ids),
            "active_probes": n_200,
            "delisted_probes": n_404,
            "error_probes": n_err,
            "after_rows": after_rows,
        }

    all_probes = [
        p for p in probe_map.values() if p.get("fsa_status") in ("active", "delisted")
    ]
    chunk_size = 500
    if len(all_probes) <= chunk_size:
        staging_sql = build_staging_table_sql(staging_path, list(probe_map.values()), today_str)
        _run_query(client, project_id, staging_sql)
    else:
        first_chunk = all_probes[:chunk_size]
        _run_query(client, project_id, build_staging_table_sql(staging_path, first_chunk, today_str))
        for offset in range(chunk_size, len(all_probes), chunk_size):
            chunk = all_probes[offset : offset + chunk_size]
            _run_query(client, project_id, build_staging_insert_sql(staging_path, chunk, today_str))

    merge_sql = build_liveness_merge_sql(bq_path, staging_path, today_str)
    _run_query(client, project_id, merge_sql)

    scoped_ids = (
        target_ids
        if (only_duplicates or only_non_standard_ratings or fhrsids) and len(target_ids) <= chunk_size
        else None
    )
    link_sql = build_link_superseded_sql(bq_path, new_ids=scoped_ids)
    _run_query(client, project_id, link_sql)

    carry_sql = build_carry_over_sql(bq_path, new_ids=scoped_ids)
    _run_query(client, project_id, carry_sql)

    # Also backfill coordinates from staging onto active rows that still lack latitude/longitude
    coord_fill_sql = f"""
    MERGE `{bq_path}` AS T
    USING `{staging_path}` AS S
    ON T.fhrsid = S.fhrsid AND T.fsa_status = 'active' AND S.fsa_status = 'active'
    WHEN MATCHED AND (T.latitude IS NULL OR T.longitude IS NULL)
         AND S.latitude IS NOT NULL AND S.longitude IS NOT NULL THEN
      UPDATE SET
        latitude = COALESCE(T.latitude, S.latitude),
        longitude = COALESCE(T.longitude, S.longitude)
    """
    _run_query(client, project_id, coord_fill_sql)

    if target_ids and len(target_ids) <= chunk_size:
        after_sql = build_candidates_query(bq_path=bq_path, fhrsids=target_ids)
    elif target_ids:
        after_sql = f"""
        SELECT *
        FROM `{bq_path}`
        WHERE fhrsid IN (SELECT fhrsid FROM `{staging_path}`)
        ORDER BY businessname, postcode, first_seen, fhrsid
        """
    else:
        after_sql = cand_sql
    after_rows = [dict(r) for r in _run_query(client, project_id, after_sql)]
    print_reconciliation_report(before_rows, after_rows, mode_label="EXECUTED")

    table_totals_sql = f"""
    SELECT
      COUNT(*) AS total_rows,
      COUNTIF(fsa_status = 'active') AS active_rows,
      COUNTIF(fsa_status = 'delisted') AS delisted_rows,
      COUNTIF(superseded_by IS NOT NULL) AS superseded_rows
    FROM `{bq_path}`
    """
    totals = [dict(r) for r in _run_query(client, project_id, table_totals_sql)]
    if totals:
        print(f"\nFull Table Totals (`{bq_path}`): {totals[0]}")

    return {
        "mode": "executed",
        "probed": len(target_ids),
        "active_probes": n_200,
        "delisted_probes": n_404,
        "error_probes": n_err,
        "after_rows": after_rows,
        "table_totals": totals[0] if totals else {},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile FSA liveness and link superseded FHRSIDs.")
    parser.add_argument("--bq_path", default=DEFAULT_BQ_PATH, help="Full BigQuery table path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview probe results and reconciliation plan without modifying BigQuery (default).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write probe results to staging and execute MERGE + supersession in BigQuery.",
    )
    parser.add_argument(
        "--only-duplicates",
        action="store_true",
        help="Only check rows belonging to name+postcode duplicate groups.",
    )
    parser.add_argument(
        "--only-non-standard-ratings",
        action="store_true",
        help="Only check active rows whose ratingvalue is NULL or not in ('1','2','3','4','5') (e.g. '0', 'AwaitingInspection', 'Exempt').",
    )
    parser.add_argument(
        "--fhrsids",
        default=None,
        help="Optional comma-separated list of FHRSIDs to probe and reconcile.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent threads for FSA API probing (default: 1).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Sleep seconds between sequential probes when max_workers=1 (default: 0.2).",
    )
    args = parser.parse_args(argv)

    fhrsids_list: Optional[List[str]] = None
    if args.fhrsids:
        fhrsids_list = [x.strip() for x in args.fhrsids.split(",") if x.strip()]

    reconcile_fsa_liveness(
        bq_path=args.bq_path,
        execute=bool(args.execute),
        only_duplicates=bool(args.only_duplicates),
        only_non_standard_ratings=bool(args.only_non_standard_ratings),
        fhrsids=fhrsids_list,
        max_workers=int(args.max_workers),
        sleep_seconds=float(args.sleep_seconds),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
