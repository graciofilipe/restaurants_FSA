import os
import time
from typing import Callable, List, Optional
from google.cloud import bigquery
import requests

from app.core.profile_freshness import (
    PreFlightEnrichmentError,
    max_allowed_enrichment_failures,
)

BQ_PATH = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

MAPS_HTTP_TIMEOUT_SECONDS = 10
MAPS_BQ_TIMEOUT_SECONDS = 60
MAPS_PROGRESS_EVERY = 10
MAPS_MERGE_BATCH_SIZE = 50

PL_MAP = {
    "PRICE_LEVEL_FREE": 0, "PRICE_LEVEL_INEXPENSIVE": 1, "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3, "PRICE_LEVEL_VERY_EXPENSIVE": 4
}


def _run_bq_query(client: bigquery.Client, sql: str, timeout: float):
    job = client.query(sql)
    try:
        return job.result(timeout=timeout)
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def _flush_maps_updates(
    client: bigquery.Client,
    table_ref: str,
    batch: list,
    bq_timeout: float,
    raise_on_error: bool,
) -> None:
    if not batch:
        return
    val_strs = []
    for u in batch:
        fid = u["fhrsid"].replace("'", "\\'")
        pl = u["price_level"] or "NULL"
        mr = u["maps_rating"] if u["maps_rating"] is not None else "NULL"
        mrev = u["maps_reviews"] if u["maps_reviews"] is not None else "NULL"
        lat = u["latitude"] if u["latitude"] is not None else "NULL"
        lon = u["longitude"] if u["longitude"] is not None else "NULL"
        murl = f"'{u['maps_url'].replace(chr(39), chr(92)+chr(39))}'" if u["maps_url"] else "NULL"
        bstat = f"'{u['business_status'].replace(chr(39), chr(92)+chr(39))}'" if u["business_status"] else "NULL"
        wurl = f"'{u['website_url'].replace(chr(39), chr(92)+chr(39))}'" if u["website_url"] else "NULL"
        mtypes = f"'{u['maps_types'].replace(chr(39), chr(92)+chr(39))}'" if u["maps_types"] else "NULL"
        found = "TRUE" if u["maps_found"] else "FALSE"
        val_strs.append(f"('{fid}', {pl}, {mr}, {mrev}, {lat}, {lon}, {murl}, {bstat}, {wurl}, {mtypes}, {found})")

    merge_q = f"""
            MERGE `{table_ref}` T
            USING (SELECT * FROM UNNEST([STRUCT<fhrsid STRING, price_level INT64, maps_rating FLOAT64, maps_reviews INT64, latitude FLOAT64, longitude FLOAT64, maps_url STRING, business_status STRING, website_url STRING, maps_types STRING, maps_found BOOL> {", ".join(val_strs)}])) S
            ON T.fhrsid = S.fhrsid
            WHEN MATCHED THEN UPDATE SET price_level=S.price_level, maps_rating=S.maps_rating, maps_reviews=S.maps_reviews, latitude=IFNULL(S.latitude, T.latitude), longitude=IFNULL(S.longitude, T.longitude), maps_url=S.maps_url, business_status=S.business_status, website_url=S.website_url, maps_types=S.maps_types, maps_found=S.maps_found, maps_lookup_at=CURRENT_TIMESTAMP()
            """
    try:
        _run_bq_query(client, merge_q, timeout=bq_timeout)
    except Exception as e:
        print(f"Merge error: {e}")
        if raise_on_error:
            raise PreFlightEnrichmentError(f"Maps BigQuery MERGE failed or timed out: {e}") from e


def enrich_restaurants_by_fhrsid(
    fhrsids: Optional[List[str]] = None,
    limit: int = 1000,
    force_regen: bool = False,
    *,
    request_timeout: float = MAPS_HTTP_TIMEOUT_SECONDS,
    bq_timeout: float = MAPS_BQ_TIMEOUT_SECONDS,
    raise_on_error: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> int:
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    if fhrsids:
        escaped_ids = [f"'{str(f).replace(chr(39), chr(39)+chr(39))}'" for f in fhrsids]
        formatted_ids = ", ".join(escaped_ids)
        fhrsid_filter = f"AND fhrsid IN ({formatted_ids})"
    else:
        fhrsid_filter = ""
    # `maps_lookup_at IS NULL`, not `maps_rating IS NULL`: a row Places has
    # already failed to find has no rating either, and the old predicate put it
    # back in the queue on every run. That is what the `-1` sentinel was for,
    # and Phase 5 retired it in favour of a timestamp that says what it means.
    null_filter = "" if force_regen else "AND maps_lookup_at IS NULL"

    query = f"SELECT fhrsid, BusinessName, PostCode, AddressLine1 FROM `{table_ref}` WHERE BusinessName IS NOT NULL {null_filter} {fhrsid_filter} LIMIT {limit}"
    try:
        rows_to_update = list(_run_bq_query(client, query, timeout=bq_timeout))
    except Exception as e:
        print(f"Error fetching from BQ: {e}")
        if raise_on_error:
            raise PreFlightEnrichmentError(f"Maps BigQuery SELECT failed or timed out: {e}") from e
        return 0

    if not rows_to_update:
        return 0

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json", "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.priceLevel,places.rating,places.userRatingCount,places.location,places.googleMapsUri,places.businessStatus,places.types,places.websiteUri"
    }

    total_rows = len(rows_to_update)
    pending_updates = []
    total_updated = 0
    total_merged = 0
    found_count = 0
    miss_count = 0
    http_failures = 0
    last_http_error: Optional[str] = None

    for idx, row in enumerate(rows_to_update, start=1):
        search_query = f"{row.BusinessName} {row.PostCode}" if row.PostCode else f"{row.BusinessName} {row.AddressLine1}"
        latest_label = f"{row.BusinessName}"
        try:
            resp = requests.post(
                url,
                json={"textQuery": search_query},
                headers=headers,
                timeout=request_timeout,
            )
            if resp.status_code >= 400:
                http_failures += 1
                last_http_error = f"HTTP {resp.status_code}"
                latest_label = f"{row.BusinessName} (HTTP {resp.status_code})"
                print(f"Error fetching for {row.BusinessName}: HTTP {resp.status_code}")
            elif resp.status_code == 200 and "places" in resp.json() and resp.json()["places"]:
                p = resp.json()["places"][0]
                pr = p.get("priceLevel")
                pl = PL_MAP.get(pr, pr) if isinstance(pr, (str, int)) else None
                loc = p.get("location", {})
                rating_val = p.get("rating")
                pending_updates.append({
                    "fhrsid": row.fhrsid, "price_level": pl, "maps_rating": rating_val, "maps_reviews": p.get("userRatingCount"),
                    "latitude": loc.get("latitude"), "longitude": loc.get("longitude"), "maps_url": p.get("googleMapsUri"),
                    "business_status": p.get("businessStatus"), "website_url": p.get("websiteUri"),
                    "maps_types": ",".join(p.get("types", [])) if p.get("types") else None,
                    "maps_found": True
                })
                total_updated += 1
                found_count += 1
                rating_str = f"{rating_val}★" if rating_val is not None else "no rating"
                latest_label = f"{row.BusinessName} ({rating_str})"
            else:
                # A miss leaves the rating NULL and says so in `maps_found`. It
                # used to write -1, which scored these worse than an unknown
                # restaurant in `calculate_restaurant_priority` while also
                # standing in for the do-not-retry flag. `maps_lookup_at` now
                # carries that second job on its own.
                pending_updates.append({"fhrsid": row.fhrsid, "price_level": None, "maps_rating": None, "maps_reviews": None, "latitude": None, "longitude": None, "maps_url": None, "business_status": None, "website_url": None, "maps_types": None, "maps_found": False})
                total_updated += 1
                miss_count += 1
                latest_label = f"{row.BusinessName} (not found)"
        except Exception as e:
            http_failures += 1
            last_http_error = str(e)
            latest_label = f"{row.BusinessName} (error)"
            print(f"Error fetching for {row.BusinessName}: {e}")

        if progress_callback and (idx % MAPS_PROGRESS_EVERY == 0 or idx == total_rows):
            pct = int(round(100 * idx / total_rows))
            progress_callback(
                f"🗺️ Maps lookup {idx}/{total_rows} ({pct}%): {found_count} found, "
                f"{miss_count} not found — latest: {latest_label}"
            )

        if len(pending_updates) >= MAPS_MERGE_BATCH_SIZE:
            _flush_maps_updates(client, table_ref, pending_updates, bq_timeout, raise_on_error)
            total_merged += len(pending_updates)
            pending_updates = []
            if progress_callback:
                progress_callback(f"💾 Merged Maps batch to BigQuery ({total_merged}/{total_rows} complete)")

        time.sleep(0.05)

    if pending_updates:
        _flush_maps_updates(client, table_ref, pending_updates, bq_timeout, raise_on_error)
        total_merged += len(pending_updates)
        pending_updates = []
        if progress_callback:
            progress_callback(f"💾 Merged Maps batch to BigQuery ({total_merged}/{total_rows} complete)")

    allowed_failures = max_allowed_enrichment_failures(len(rows_to_update))
    if raise_on_error and http_failures > allowed_failures:
        raise PreFlightEnrichmentError(
            f"Maps Places API failed for {http_failures}/{len(rows_to_update)} restaurants "
            f"(allowed <= {allowed_failures}; last error: {last_http_error})."
        )

    return total_updated

if __name__ == "__main__":
    enrich_restaurants_by_fhrsid()

