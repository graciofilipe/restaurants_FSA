import datetime
import logging
import math
import os
from google.cloud import bigquery
import pandas as pd
from typing import List, Set, Tuple

from app.core.data_processing import (
    normalize_fhrsid,
    parse_coordinates,
    fetch_data_for_all_coordinates,
    process_and_update_master_data,
    parse_bq_path
)
from app.services.bq_utils import (
    BigQueryExecutionError,
    load_fhrsids_from_bq,
    append_to_bigquery,
    touch_seen_fhrsids,
    candidates_for_liveness_check,
    mark_fhrsids_delisted,
    link_superseded,
    MASTER_BQ_SCHEMA,
    ORIGINAL_COLUMNS_TO_KEEP
)
from scripts.reconcile_fsa_liveness import probe_establishments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config details
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "filipegracio-ai-learning")
DATASET_ID = "filipegracio_fsa_restaurants"
CONFIG_TABLE_ID = "config_search_params"


def compute_area_bbox(lat: float, lon: float, radius_km: float = 5.0) -> Tuple[float, float, float, float]:
    """Compute `(min_lat, max_lat, min_lon, max_lon)` around `(lat, lon)` for `radius_km`."""
    lat_delta = float(radius_km) / 111.0
    cos_lat = max(0.1, abs(math.cos(math.radians(float(lat)))))
    lon_delta = float(radius_km) / (111.0 * cos_lat)
    return (
        float(lat) - lat_delta,
        float(lat) + lat_delta,
        float(lon) - lon_delta,
        float(lon) + lon_delta,
    )


def _unpack_process_result(res, all_api_establishments) -> Tuple[List[dict], Set[str], str]:
    """Accept either `(new_records, summary_msg)` (with optional `.seen_ids`) or `(new_records, seen_ids, summary_msg)`."""
    if isinstance(res, tuple) and len(res) == 3:
        new_restaurants, seen_ids, summary_msg = res
        return new_restaurants, set(seen_ids or ()), summary_msg
    new_restaurants, summary_msg = res
    seen_ids = getattr(res, "seen_ids", None)
    if seen_ids is None:
        seen_ids = {
            normalize_fhrsid(e["FHRSID"])
            for e in (all_api_establishments or [])
            if isinstance(e, dict) and e.get("FHRSID") is not None
        }
    return new_restaurants, set(seen_ids), summary_msg


def get_config_params() -> List[dict]:
    """Reads configuration parameters from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{CONFIG_TABLE_ID}"
    query = f"SELECT * FROM `{table_ref}`"
    
    logger.info(f"Reading config from {table_ref}")
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        config_list = []
        for row in results:
            config_list.append(dict(row))
        return config_list
    except Exception as e:
        logger.error(f"Error reading config table: {e}")
        return []

def run_sync_for_config(config: dict):
    """Runs synchronization for a single configuration row."""
    lat = config.get('latitude')
    lon = config.get('longitude')
    max_results = config.get('max_results')
    target_bq_path = config.get('target_bq_table')
    
    logger.info(f"Starting sync for target: {target_bq_path} with max_results: {max_results}")
    
    try:
        project_id, dataset_id, table_id = parse_bq_path(target_bq_path)
    except ValueError as e:
        logger.error(f"Invalid target BQ path: {target_bq_path}. Error: {e}")
        return

    # 1. Parse/Validate Coordinates
    if lat is None or lon is None:
        logger.error(f"Missing latitude or longitude in config: {config}")
        return

    try:
        # fetch_data_for_all_coordinates expects (lon, lat)
        valid_coords = [(float(lon), float(lat))]
    except ValueError as e:
        logger.error(f"Invalid coordinate values: lat={lat}, lon={lon}. Error: {e}")
        return

    # 2. Fetch API Data
    logger.info("Fetching data from API...")
    all_api_establishments = fetch_data_for_all_coordinates(valid_coords, max_results)
    logger.info(f"Fetched {len(all_api_establishments)} records from API.")
    
    combined_api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': all_api_establishments}}}

    # 3. Load the IDs we already hold -- nothing else is needed to spot new records
    logger.info("Loading existing FHRSIDs from BigQuery...")
    # Deliberately not caught. Aborting is right -- appending without knowing
    # what already exists would duplicate the table -- but swallowing it left
    # `main()` reporting a successful week in which nothing was ingested (D9).
    existing_fhrsids = load_fhrsids_from_bq(project_id, dataset_id, table_id)
    logger.info(f"Loaded {len(existing_fhrsids)} existing records.")

    # 4. Process and Identify New + Seen IDs
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    logger.info("Processing data to identify new records...")
    proc_res = process_and_update_master_data(existing_fhrsids, combined_api_data)
    new_restaurants, seen_ids, summary_msg = _unpack_process_result(proc_res, all_api_establishments)
    logger.info(f"Process summary: {summary_msg}")

    new_ids = [
        str(r.get("FHRSID") or r.get("fhrsid"))
        for r in new_restaurants
        if (r.get("FHRSID") or r.get("fhrsid")) is not None
    ]

    if new_restaurants:
        # 5. Append to BigQuery
        logger.info(f"Appending {len(new_restaurants)} new records to BigQuery...")
        df_new = pd.DataFrame(new_restaurants)

        # Normalize columns to lowercase to match BQ schema
        df_new.columns = [c.lower() for c in df_new.columns]

        # Ensure columns match schema
        success = append_to_bigquery(
            df=df_new,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
            bq_schema=MASTER_BQ_SCHEMA
        )

        if not success:
            # `append_to_bigquery` reports failure by returning False. Logging it
            # and returning meant the new restaurants were dropped on the floor and
            # the run still counted as a success.
            raise BigQueryExecutionError(
                f"Failed to append {len(new_restaurants)} new records to {table_id}")
        logger.info("Append successful.")
    else:
        logger.info("No new restaurants to append.")

    # 6. Liveness reconciliation for this search area when API returned establishments
    if seen_ids:
        api_rating_updates = {}
        for est in all_api_establishments or []:
            if isinstance(est, dict) and est.get("FHRSID") is not None:
                fid_norm = normalize_fhrsid(est["FHRSID"])
                rv = est.get("RatingValue")
                nrp = est.get("NewRatingPending")
                if rv is not None or nrp is not None:
                    api_rating_updates[fid_norm] = {
                        "ratingvalue": rv,
                        "newratingpending": nrp,
                    }
        touch_kwargs = {"today": today_str}
        if api_rating_updates:
            touch_kwargs["rating_updates"] = api_rating_updates
        touched = touch_seen_fhrsids(project_id, dataset_id, table_id, seen_ids, **touch_kwargs)
        logger.info(f"Touched last_seen={today_str} on {touched} active establishments.")

        radius_km = float(config.get("radius") or 5.0)
        area_bbox = compute_area_bbox(float(lat), float(lon), radius_km=radius_km)
        unseen_candidates = candidates_for_liveness_check(
            project_id, dataset_id, table_id, area_bbox, today=today_str
        )
        newly_delisted: List[str] = []
        still_live: List[str] = []
        probe_rating_updates = {}
        if unseen_candidates:
            logger.info(f"Probing {len(unseen_candidates)} unseen active candidate(s) in search bbox...")
            probes = probe_establishments(unseen_candidates)
            for fid, info in probes.items():
                status = info.get("status_code") or info.get("http_status")
                if status == 404:
                    newly_delisted.append(fid)
                elif status == 200:
                    still_live.append(fid)
                    rv = info.get("ratingvalue")
                    nrp = info.get("newratingpending")
                    if rv is not None or nrp is not None:
                        probe_rating_updates[fid] = {
                            "ratingvalue": rv,
                            "newratingpending": nrp,
                        }
            if still_live:
                live_kwargs = {"today": today_str}
                if probe_rating_updates:
                    live_kwargs["rating_updates"] = probe_rating_updates
                touch_seen_fhrsids(project_id, dataset_id, table_id, still_live, **live_kwargs)
            if newly_delisted:
                delisted_cnt = mark_fhrsids_delisted(
                    project_id, dataset_id, table_id, newly_delisted, today=today_str
                )
                logger.info(f"Marked {delisted_cnt} establishment(s) delisted (404 on FSA API).")

        if new_ids or newly_delisted:
            linked = link_superseded(
                project_id, dataset_id, table_id, new_ids=new_ids if new_ids else None
            )
            logger.info(f"Linked {linked} superseded establishment(s) to active successors.")

def main():
    """Run every configured search area, and tell the truth about the result.

    Exit status is the only thing Cloud Run reads. This used to catch
    everything and return, so the job exited 0 whatever happened -- a weekly
    ingest could fail on an expired credential for a month, and the only
    evidence would be log lines nobody opens and a `first_seen` gap nobody
    would attribute to this (D9).

    One bad search area still does not cost the others their run; the failures
    are collected and reported together at the end.
    """
    logger.info("Starting Weekly Fetch Job")
    try:
        configs = get_config_params()
    except Exception as e:
        logger.exception(f"Could not read the search configuration: {e}")
        raise SystemExit(f"Weekly fetch aborted: could not read the search configuration: {e}")

    if not configs:
        # Not a failure. Emptying the config table is how the cron gets paused,
        # and a paused job that fails every week would just get muted.
        logger.warning("No configuration found in config table.")
        return

    failures = []
    for config in configs:
        try:
            run_sync_for_config(config)
        except Exception as e:
            logger.exception(f"Error processing config {config}: {e}")
            failures.append(e)

    if failures:
        raise SystemExit(
            f"Weekly fetch: {len(failures)} of {len(configs)} search areas failed. "
            f"First error: {failures[0]}")

    logger.info(f"Weekly Fetch Job complete: {len(configs)} search areas processed.")

if __name__ == "__main__":
    main()
