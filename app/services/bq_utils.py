import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, exceptions as google_cloud_exceptions
import pandas as pd
from app.core.pillar_schema import (
    NON_JSON_COLUMNS,
    PILLAR_FIELDS,
    sql_conformance_check,
    summarise_conformance,
)
from app.core.profile_freshness import max_allowed_enrichment_failures
from scripts.bq_scripts import (
    MODEL_PARAMS_JSON,
    SCRIPT_BULK_UPDATE_MERGE,
    SCRIPT_GENERATE_INSIGHTS,
    SCRIPT_IDENTIFY_RECENTS,
    SCRIPT_MERGE_INSIGHTS,
)

logger = logging.getLogger(__name__)

GEMINI_ENRICHMENT_BATCH_SIZE = 25
GEMINI_BQ_TIMEOUT_SECONDS = 180.0

# The fields the weekly ingest copies out of an FSA establishment. Flat keys
# only -- `process_and_update_master_data` copies by key, so `latitude` and
# `longitude` are the flattened form of the API's nested `Geocode`, written by
# `extract_fsa_coordinates` before the copy.
ORIGINAL_COLUMNS_TO_KEEP = [
    'FHRSID', 'BusinessName', 'AddressLine1', 'AddressLine2', 'AddressLine3',
    'PostCode', 'LocalAuthorityName', 'RatingValue', 'NewRatingPending',
    'latitude', 'longitude',
    'first_seen', 'gemini_insights_structured'
]

class BigQueryExecutionError(Exception):
    pass

class DataFrameConversionError(Exception):
    pass

FHRSID_COLNAME = "fhrsid"

def _sql_quote(val: Any) -> str:
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _wait_for_job(job: Any, timeout: Optional[float] = GEMINI_BQ_TIMEOUT_SECONDS) -> Any:
    """Wait for a BigQuery job with a hard timeout, cancelling the job if it stalls."""
    try:
        if timeout is not None:
            return job.result(timeout=timeout)
        return job.result()
    except Exception:
        try:
            job.cancel()
        except Exception:
            pass
        raise


def log_insight_conformance(
    client: "bigquery.Client", project_id: str, dataset_id: str, insights_table_id: str
) -> Optional[Dict[str, Any]]:
    """Report how far this run's output strays from the canonical pillar shape.

    Runs against the scratch table *before* the merge, so the numbers describe
    what was just generated rather than the whole accumulated column.

    Deliberately advisory: it logs and returns, and never raises or blocks the
    merge. The evidence in D-08 is that the profiler conforms on 2,766 of 2,767
    rows, so refusing to merge on a single bad path would throw away good
    profiles over a leaked reasoning trace. What was missing was not
    enforcement but any way at all to notice -- five features read zero for the
    life of the model and nothing said so.
    """
    table_ref = f"{project_id}.{dataset_id}.{insights_table_id}"
    try:
        rows = list(client.query(sql_conformance_check(table_ref)).result(timeout=60))
        if not rows:
            return None
        row = dict(rows[0])
        total, offenders = summarise_conformance(row)
        if offenders:
            detail = ", ".join(f"{name}={count}" for name, count in sorted(offenders.items()))
            logger.warning(
                f"Profile conformance: {total} generated, non-conforming paths -- {detail}. "
                f"The pillar schema may have drifted; see app/core/pillar_schema.py."
            )
        else:
            logger.info(f"Profile conformance: {total} generated, all paths present.")
        return row
    except Exception as e:
        # A failed check must not cost the run the profiles it just paid for.
        logger.warning(f"Could not run the profile conformance check: {e}")
        return None


def execute_gemini_enrichment(
    project_id: str,
    dataset_id: str,
    master_table_id: str,
    connection_id: str = 'eu.gemini',
    model_endpoint: str = 'gemini-3.8-flash',
    days_recent: int = 33,
    excluded_locations: Optional[List[str]] = None,
    fhrsids: Optional[List[str]] = None,
    *,
    batch_size: int = GEMINI_ENRICHMENT_BATCH_SIZE,
    query_timeout: Optional[float] = GEMINI_BQ_TIMEOUT_SECONDS,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Orchestrates the Gemini enrichment process using BigQuery SQL scripts.

    When `fhrsids` exceeds `batch_size`, chunks execution into bounded batches
    with a per-step `query_timeout` (cancelling stalled BigQuery jobs) and
    verifies that at least 95% of targeted rows were merged.
    """
    client = bigquery.Client(project=project_id)
    effective_batch_size = max(1, int(batch_size))
    if fhrsids:
        batches: List[Optional[List[str]]] = [
            fhrsids[i : i + effective_batch_size]
            for i in range(0, len(fhrsids), effective_batch_size)
        ]
    else:
        batches = [None]

    total_merged = 0
    tracked_dml_counts = False

    for batch_idx, batch_ids in enumerate(batches, start=1):
        # Per-run names: these scratch tables sit in the production dataset, and two
        # overlapping runs sharing `recents` would each profile the other's selection.
        run_id = uuid.uuid4().hex[:12]
        recents_table_id, insights_table_id = f"recents_{run_id}", f"genairesults_temp_{run_id}"
        batch_start_ts = time.monotonic()
        try:
            if batch_ids is not None:
                start_row = (batch_idx - 1) * effective_batch_size + 1
                end_row = start_row + len(batch_ids) - 1
                total_target = len(fhrsids) if fhrsids else len(batch_ids)
                if progress_callback:
                    progress_callback(
                        f"✨ Regenerating Gemini profiles: batch {batch_idx}/{len(batches)} "
                        f"({len(batch_ids)} restaurant(s), rows {start_row}–{end_row} of {total_target})..."
                    )
                escaped = [_sql_quote(f) for f in batch_ids]
                filter_condition = f"CAST(fhrsid AS STRING) IN ({', '.join(escaped)})"
            else:
                if progress_callback:
                    progress_callback("✨ Running Gemini enrichment for recent restaurants...")
                excl_clause = ""
                if excluded_locations:
                    escaped_locs = [_sql_quote(l) for l in excluded_locations]
                    excl_clause = f"AND localauthorityname NOT IN ({', '.join(escaped_locs)})"
                # `in_scope IS NOT FALSE`, not `IS TRUE`: every row arrives untriaged
                # and profiling is usually what decides the question, so `IS TRUE`
                # would mean a new restaurant is never looked at. What this does
                # exclude is the already-answered no -- a cafe or a bakery, judged
                # by triage or by an earlier profile, which there is no reason to
                # pay to profile again. It replaces `manual_review IN (...)`, a
                # free-text column whose dominant value is 'rejected' on 9,348 rows
                # that are in scope.
                filter_condition = f"DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {days_recent} AND in_scope IS NOT FALSE {excl_clause}"

            q_recents = SCRIPT_IDENTIFY_RECENTS.format(
                project_id=project_id, dataset_id=dataset_id, source_table=master_table_id,
                target_table_recents=recents_table_id, filter_condition=filter_condition
            )
            _wait_for_job(client.query(q_recents), timeout=query_timeout)

            q_insights = SCRIPT_GENERATE_INSIGHTS.format(
                project_id=project_id, dataset_id=dataset_id, source_table_recents=recents_table_id,
                target_table_insights=insights_table_id, connection_id=connection_id,
                model_endpoint=model_endpoint, model_params_json=MODEL_PARAMS_JSON
            )
            _wait_for_job(client.query(q_insights), timeout=query_timeout)

            log_insight_conformance(client, project_id, dataset_id, insights_table_id)

            q_merge = SCRIPT_MERGE_INSIGHTS.format(
                project_id=project_id, dataset_id=dataset_id, source_table_insights=insights_table_id,
                target_table_master=master_table_id
            )
            job = client.query(q_merge)
            _wait_for_job(job, timeout=query_timeout)
            dml_rows = getattr(job, "num_dml_affected_rows", None)
            if isinstance(dml_rows, int):
                tracked_dml_counts = True
                total_merged += dml_rows
                merged_in_batch = dml_rows
            else:
                merged_in_batch = len(batch_ids) if batch_ids is not None else 0
            batch_elapsed = time.monotonic() - batch_start_ts
            if progress_callback and batch_ids is not None:
                total_target = len(fhrsids) if fhrsids else len(batch_ids)
                done_count = total_merged if tracked_dml_counts else min(total_target, batch_idx * effective_batch_size)
                progress_callback(
                    f"✅ Gemini batch {batch_idx}/{len(batches)} merged: {merged_in_batch} row(s) updated in {batch_elapsed:.1f}s ({done_count}/{total_target} complete)"
                )
        except Exception as e:
            logger.error(f"Error during Gemini enrichment (batch {batch_idx}/{len(batches)}): {e}")
            return False
        finally:
            for temp_table_id in (recents_table_id, insights_table_id):
                try:
                    client.delete_table(f"{project_id}.{dataset_id}.{temp_table_id}", not_found_ok=True)
                except Exception as e:
                    # The table expires on its own; a failed drop is not worth failing the run.
                    logger.warning(f"Could not drop temp table {temp_table_id}: {e}")

    if fhrsids and tracked_dml_counts:
        allowed_failures = max_allowed_enrichment_failures(len(fhrsids))
        shortfall = len(fhrsids) - total_merged
        if shortfall > allowed_failures:
            logger.error(
                f"Gemini enrichment merged only {total_merged}/{len(fhrsids)} profiles "
                f"({shortfall} failed/NULL, allowed <= {allowed_failures})."
            )
            return False

    return True

def load_fhrsids_from_bq(project_id: str, dataset_id: str, table_id: str) -> Set[str]:
    """Loads just the FHRSIDs from a table, for deduplicating an ingest.

    Unlike the loaders around it this one raises: the caller treats the result
    as "everything that already exists", so an empty set from a failed read
    would re-append the whole fetch.
    """
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client = bigquery.Client(project=project_id)
        results = client.query(f"SELECT fhrsid FROM `{table_ref}`").result()
        return {str(row.fhrsid) for row in results if row.fhrsid is not None}
    except Exception as e:
        logger.error(f"Error loading FHRSIDs from {table_ref}: {e}")
        raise BigQueryExecutionError(f"Could not load FHRSIDs from {table_ref}: {e}") from e

def load_filtered_data_from_bq(
    project_id: str,
    dataset_id: str,
    table_id: str,
    days_filter: Optional[int] = None,
    excluded_locations: Optional[List[str]] = None,
    postcode_areas: Optional[List[str]] = None,
    first_seen_start_date: Optional[str] = None,
    local_authority_filter: Optional[List[str]] = None,
    in_scope_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Loads filtered restaurant data from BigQuery."""
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref}` WHERE 1=1"

    if days_filter is not None:
        query += f" AND DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {days_filter}"
    if first_seen_start_date:
        query += f" AND first_seen >= '{first_seen_start_date}'"
    if in_scope_filter:
        scope_clauses = []
        if 'in_scope' in in_scope_filter:
            scope_clauses.append("in_scope = TRUE")
        if 'out_of_scope' in in_scope_filter:
            scope_clauses.append("in_scope = FALSE")
        if 'unprocessed' in in_scope_filter:
            scope_clauses.append("in_scope IS NULL")
        if scope_clauses:
            query += f" AND ({' OR '.join(scope_clauses)})"
    if local_authority_filter:
        escaped = [_sql_quote(a) for a in local_authority_filter]
        query += f" AND localauthorityname IN ({', '.join(escaped)})"
    if excluded_locations:
        escaped = [_sql_quote(l) for l in excluded_locations]
        query += f" AND localauthorityname NOT IN ({', '.join(escaped)})"
    if postcode_areas:
        escaped = [_sql_quote(p) for p in postcode_areas]
        query += f" AND SPLIT(postcode, ' ')[SAFE_OFFSET(0)] IN ({', '.join(escaped)})"

    try:
        client = bigquery.Client(project=project_id)
        results = client.query(query).result()
        records = []
        for row in results:
            rec = dict(row)
            if rec.get('first_seen') is not None:
                rec['first_seen'] = str(rec['first_seen'])
            if rec.get('predicted_at') is not None:
                rec['predicted_at'] = str(rec['predicted_at'])
            records.append(rec)
        return records
    except Exception as e:
        logger.error(f"Error loading filtered data from {table_ref}: {e}")
        # Not `return []`. An empty list is what "nothing matched your filters"
        # looks like, so returning it here made a dead credential render as
        # "No data found matching criteria" -- advice to widen the filters,
        # for a problem no filter can reach (D9).
        raise BigQueryExecutionError(f"Could not load data from {table_ref}: {e}") from e

def sanitize_column_name(column_name: str) -> str:
    """Sanitizes a column name for BigQuery compatibility."""
    name = column_name.replace(' ', '_').replace('.', '').replace('@', '').replace('-', '_').lower()
    if name and not name[0].isalnum() and name[0] != '_':
        name = name[1:]
    name = re.sub(r'[^a-z0-9_]+', '_', name).strip('_')
    return name or "unnamed_column"

def bulk_update_reviews(
    project_id: str, dataset_id: str, target_table_id: str, df_updates: pd.DataFrame
) -> Tuple[bool, str]:
    """Performs a bulk update of in_scope, user_rating, and/or rating_source columns using a temp table and MERGE."""
    if df_updates.empty:
        return False, "DataFrame is empty."

    df_updates = df_updates.copy()
    df_updates.columns = [col.lower() for col in df_updates.columns]
    if 'fhrsid' not in df_updates.columns:
        return False, "Missing required column 'fhrsid'."

    df_updates['fhrsid'] = df_updates['fhrsid'].astype(str).str.strip()
    df_updates = df_updates.drop_duplicates(subset=['fhrsid'], keep='last')
    df_updates = df_updates[df_updates['fhrsid'].notna() & (df_updates['fhrsid'] != '') & (df_updates['fhrsid'] != 'nan')]
    if df_updates.empty:
        return False, "No valid fhrsid values provided."

    possible_update_cols = ['user_rating', 'in_scope', 'rating_source']
    updatable_cols = [c for c in possible_update_cols if c in df_updates.columns]
    if not updatable_cols:
        return False, f"No updatable columns provided in DataFrame. Expected at least one of {possible_update_cols}"

    required_cols = ['fhrsid'] + updatable_cols

    temp_table_id = f"temp_update_reviews_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    temp_schema = [bigquery.SchemaField("fhrsid", "STRING")]
    for col in updatable_cols:
        if col == 'in_scope':
            temp_schema.append(bigquery.SchemaField("in_scope", "BOOLEAN"))
        elif col == 'user_rating':
            temp_schema.append(bigquery.SchemaField("user_rating", "INT64"))
        elif col == 'rating_source':
            temp_schema.append(bigquery.SchemaField(col, "STRING"))

    if not write_to_bigquery(df_updates, project_id, dataset_id, temp_table_id, required_cols, temp_schema):
        return False, "Failed to upload temporary table to BigQuery."

    client = bigquery.Client(project=project_id)
    try:
        clauses = [f"T.{col} = S.{col}" for col in updatable_cols]
        query = SCRIPT_BULK_UPDATE_MERGE.format(
            project_id=project_id, dataset_id=dataset_id, target_table=target_table_id,
            source_table_temp=temp_table_id, update_set_clause=', '.join(clauses)
        )
        job = client.query(query)
        job.result()
        affected = job.num_dml_affected_rows
        return True, f"{affected} rows updated."
    except Exception as e:
        logger.error(f"Error during bulk update: {e}")
        return False, f"Error executing update: {str(e)}"
    finally:
        try:
            client.delete_table(f"{project_id}.{dataset_id}.{temp_table_id}", not_found_ok=True)
        except Exception as e:
            logger.warning(f"Could not drop temp table {temp_table_id}: {e}")

def write_to_bigquery(
    df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str,
    columns_to_select: List[str], bq_schema: List[bigquery.SchemaField]
) -> bool:
    """Writes a Pandas DataFrame to a BigQuery table with WRITE_TRUNCATE."""
    for col in columns_to_select:
        if col not in df.columns:
            df[col] = pd.NA
    df_sub = df[columns_to_select].copy()

    df_sub.columns = [sanitize_column_name(c) for c in df_sub.columns]
    nrp = sanitize_column_name('NewRatingPending')
    if nrp in df_sub.columns:
        df_sub[nrp] = df_sub[nrp].astype(str).str.lower().map({'true': True, 'false': False}).fillna(pd.NA)

    if 'fhrsid' in df_sub.columns and df_sub['fhrsid'].dtype != 'object':
        df_sub['fhrsid'] = df_sub['fhrsid'].astype(str)

    try:
        client = bigquery.Client(project=project_id)
        job_config = bigquery.LoadJobConfig(schema=bq_schema, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, column_name_character_map="V2")
        client.load_table_from_dataframe(df_sub, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config).result()
        return True
    except Exception as e:
        logger.error(f"Error writing to BigQuery: {e}")
        return False

def append_to_bigquery(
    df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, bq_schema: List[bigquery.SchemaField]
) -> bool:
    """Appends a Pandas DataFrame to an existing BigQuery table."""
    schema_cols = [f.name for f in bq_schema]
    for col in schema_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df_sub = df[schema_cols].copy()

    # The FSA sends its coordinates quoted and the columns are FLOAT64. This
    # used to name `geocode_latitude`/`geocode_longitude`, which the ingest has
    # never produced -- it dropped the nested `Geocode` entirely.
    for geo in ['latitude', 'longitude']:
        if geo in df_sub.columns:
            df_sub[geo] = pd.to_numeric(df_sub[geo], errors='coerce')

    if 'newratingpending' in df_sub.columns:
        df_sub['newratingpending'] = df_sub['newratingpending'].astype(str).str.lower().map({'true': True, 'false': False}).astype('boolean')
    if 'first_seen' in df_sub.columns:
        df_sub['first_seen'] = pd.to_datetime(df_sub['first_seen'], errors='coerce').dt.date

    if 'fhrsid' in df_sub.columns:
        ftype = next((f.field_type for f in bq_schema if f.name == 'fhrsid'), None)
        if ftype in ['INTEGER', 'INT64', 'NUMERIC']:
            df_sub['fhrsid'] = pd.to_numeric(df_sub['fhrsid'], errors='coerce')
        elif ftype == 'STRING':
            df_sub['fhrsid'] = df_sub['fhrsid'].astype(str)

    try:
        client = bigquery.Client(project=project_id)
        job_config = bigquery.LoadJobConfig(schema=bq_schema, write_disposition=bigquery.WriteDisposition.WRITE_APPEND, column_name_character_map="V2")
        client.load_table_from_dataframe(df_sub, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config).result()
        return True
    except Exception as e:
        logger.error(f"Error appending to BigQuery: {e}")
        return False



def get_distinct_local_authorities(project_id: str, dataset_id: str, table_id: str) -> List[str]:
    """Fetches distinct LocalAuthorityName values from the master table."""
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client = bigquery.Client(project=project_id)
        results = client.query(f"SELECT DISTINCT localauthorityname FROM `{table_ref}` WHERE localauthorityname IS NOT NULL ORDER BY localauthorityname").result()
        return [row.localauthorityname for row in results if row.localauthorityname]
    except Exception as e:
        logger.error(f"Error fetching local authorities: {e}")
        raise BigQueryExecutionError(
            f"Could not load local authorities from {table_ref}: {e}") from e

def get_distinct_outcodes(project_id: str, dataset_id: str, table_id: str) -> List[str]:
    """Fetches distinct Postcode Areas (outcodes) from the master table."""
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT DISTINCT SPLIT(postcode, ' ')[SAFE_OFFSET(0)] as outcode FROM `{table_ref}` WHERE postcode IS NOT NULL ORDER BY outcode"
    try:
        client = bigquery.Client(project=project_id)
        results = client.query(query).result()
        return sorted([str(r.outcode).strip() for r in results if r.outcode and str(r.outcode).strip()])
    except Exception as e:
        logger.error(f"Error fetching outcodes: {e}")
        raise BigQueryExecutionError(f"Could not load outcodes from {table_ref}: {e}") from e


FEATURE_IMPORTANCE_TABLE_ID = "model_feature_importance"
LINEAR_MODEL_NAME = "restaurant_preference_model_linear"
LINEAR_WEIGHTS_TABLE_ID = "model_linear_weights"


def _to_utc_dt(val: Any) -> Optional[ Any ]:
    import datetime as _dt
    if val is None:
        return None
    if isinstance(val, _dt.datetime):
        return val.replace(tzinfo=_dt.timezone.utc) if val.tzinfo is None else val.astimezone(_dt.timezone.utc)
    if isinstance(val, str):
        try:
            parsed = _dt.datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
            return parsed.replace(tzinfo=_dt.timezone.utc) if parsed.tzinfo is None else parsed.astimezone(_dt.timezone.utc)
        except ValueError:
            return None
    return None


def ensure_model_feature_importance(
    project_id: str,
    dataset_id: str,
    model_name: str = "restaurant_preference_model",
    model_trained_at: Any = None,
    vertex_version: Optional[str] = None,
    client: Optional["bigquery.Client"] = None,
) -> List[Dict[str, Any]]:
    """Return the latest model's feature importance, generating and saving it in
    BigQuery (`model_feature_importance`) at most once per trained model version.
    """
    bq_client = client or bigquery.Client(project=project_id)
    fi_table_ref = f"{project_id}.{dataset_id}.{FEATURE_IMPORTANCE_TABLE_ID}"
    model_ref = f"{project_id}.{dataset_id}.{model_name}"
    expected_dt = _to_utc_dt(model_trained_at)

    read_sql = f"""
    SELECT
      model_name,
      model_trained_at,
      vertex_version,
      feature,
      importance_weight,
      importance_gain,
      importance_cover,
      gain_pct,
      computed_at
    FROM `{fi_table_ref}`
    ORDER BY importance_gain DESC
    """
    try:
        existing_rows = [dict(r) for r in _wait_for_job(bq_client.query(read_sql), timeout=30.0)]
        if existing_rows:
            stored_dt = _to_utc_dt(existing_rows[0].get("model_trained_at"))
            if expected_dt is None or (
                stored_dt is not None and abs((expected_dt - stored_dt).total_seconds()) < 2.0
            ):
                return existing_rows
    except Exception:
        # Table does not exist yet or schema changed; regenerate below.
        pass

    trained_iso = expected_dt.strftime("%Y-%m-%d %H:%M:%S+00:00") if expected_dt else "1970-01-01 00:00:00+00:00"
    ver_literal = _sql_quote(vertex_version or "")
    model_literal = _sql_quote(model_name)

    materialize_sql = f"""
    CREATE OR REPLACE TABLE `{fi_table_ref}` AS
    SELECT
      {model_literal} AS model_name,
      TIMESTAMP('{trained_iso}') AS model_trained_at,
      {ver_literal} AS vertex_version,
      CAST(feature AS STRING) AS feature,
      CAST(importance_weight AS INT64) AS importance_weight,
      CAST(importance_gain AS FLOAT64) AS importance_gain,
      CAST(importance_cover AS FLOAT64) AS importance_cover,
      ROUND(100.0 * SAFE_DIVIDE(importance_gain, SUM(importance_gain) OVER ()), 2) AS gain_pct,
      CURRENT_TIMESTAMP() AS computed_at
    FROM ML.FEATURE_IMPORTANCE(MODEL `{model_ref}`)
    ORDER BY importance_gain DESC
    """
    try:
        _wait_for_job(bq_client.query(materialize_sql), timeout=60.0)
        return [dict(r) for r in _wait_for_job(bq_client.query(read_sql), timeout=30.0)]
    except Exception as e:
        logger.warning(f"Could not materialize or read model feature importance for {model_ref}: {e}")
        return []


def ensure_model_linear_weights(
    project_id: str,
    dataset_id: str,
    linear_model_name: str = LINEAR_MODEL_NAME,
    model_trained_at: Any = None,
    client: Optional["bigquery.Client"] = None,
) -> List[Dict[str, Any]]:
    """Return the companion linear model's feature weights (`ML.WEIGHTS`),
    generating and saving them in BigQuery (`model_linear_weights`) at most
    once per trained linear model version.
    """
    bq_client = client or bigquery.Client(project=project_id)
    lw_table_ref = f"{project_id}.{dataset_id}.{LINEAR_WEIGHTS_TABLE_ID}"
    model_ref = f"{project_id}.{dataset_id}.{linear_model_name}"
    expected_dt = _to_utc_dt(model_trained_at)

    read_sql = f"""
    SELECT
      model_name,
      model_trained_at,
      feature,
      feature_type,
      standardized_weight,
      raw_weight,
      category_count,
      category_spread,
      importance_magnitude,
      top_categories,
      computed_at
    FROM `{lw_table_ref}`
    ORDER BY IF(feature_type = 'intercept', 1, 0) ASC, importance_magnitude DESC
    """
    try:
        existing_rows = [dict(r) for r in _wait_for_job(bq_client.query(read_sql), timeout=30.0)]
        if existing_rows:
            stored_dt = _to_utc_dt(existing_rows[0].get("model_trained_at"))
            if expected_dt is None or (
                stored_dt is not None and abs((expected_dt - stored_dt).total_seconds()) < 2.0
            ):
                return existing_rows
    except Exception:
        # Table does not exist yet or schema changed; regenerate below.
        pass

    trained_iso = expected_dt.strftime("%Y-%m-%d %H:%M:%S+00:00") if expected_dt else "1970-01-01 00:00:00+00:00"
    model_literal = _sql_quote(linear_model_name)

    materialize_sql = f"""
    CREATE OR REPLACE TABLE `{lw_table_ref}` AS
    WITH std_w AS (
      SELECT * FROM ML.WEIGHTS(MODEL `{model_ref}`, STRUCT(TRUE AS standardize))
    ),
    raw_w AS (
      SELECT * FROM ML.WEIGHTS(MODEL `{model_ref}`, STRUCT(FALSE AS standardize))
    )
    SELECT
      {model_literal} AS model_name,
      TIMESTAMP('{trained_iso}') AS model_trained_at,
      CAST(s.processed_input AS STRING) AS feature,
      CASE
        WHEN s.processed_input = '__INTERCEPT__' THEN 'intercept'
        WHEN ARRAY_LENGTH(s.category_weights) > 0 THEN 'categorical'
        ELSE 'numeric'
      END AS feature_type,
      ROUND(CAST(s.weight AS FLOAT64), 4) AS standardized_weight,
      ROUND(CAST(r.weight AS FLOAT64), 4) AS raw_weight,
      CAST(ARRAY_LENGTH(s.category_weights) AS INT64) AS category_count,
      ROUND(
        (
          SELECT MAX(cw.weight) - MIN(cw.weight)
          FROM UNNEST(s.category_weights) AS cw
          WHERE cw.category != '_null_filler'
        ),
        4
      ) AS category_spread,
      ROUND(
        COALESCE(
          ABS(CAST(s.weight AS FLOAT64)),
          (
            SELECT MAX(cw.weight) - MIN(cw.weight)
            FROM UNNEST(s.category_weights) AS cw
            WHERE cw.category != '_null_filler'
          ),
          0.0
        ),
        4
      ) AS importance_magnitude,
      (
        SELECT STRING_AGG(
          CONCAT(
            ranked.category,
            ' (',
            IF(ranked.weight >= 0, '+', ''),
            CAST(ROUND(ranked.weight, 2) AS STRING),
            ')'
          ),
          ', '
          ORDER BY ranked.weight DESC
        )
        FROM (
          SELECT
            cw.category,
            cw.weight,
            ROW_NUMBER() OVER (ORDER BY cw.weight DESC) AS rn_desc,
            ROW_NUMBER() OVER (ORDER BY cw.weight ASC) AS rn_asc
          FROM UNNEST(s.category_weights) AS cw
          WHERE cw.category != '_null_filler'
        ) AS ranked
        WHERE ranked.rn_desc <= 2 OR ranked.rn_asc <= 2
      ) AS top_categories,
      CURRENT_TIMESTAMP() AS computed_at
    FROM std_w AS s
    LEFT JOIN raw_w AS r
      USING (processed_input)
    ORDER BY IF(s.processed_input = '__INTERCEPT__', 1, 0) ASC, importance_magnitude DESC
    """
    try:
        _wait_for_job(bq_client.query(materialize_sql), timeout=60.0)
        return [dict(r) for r in _wait_for_job(bq_client.query(read_sql), timeout=30.0)]
    except Exception as e:
        logger.warning(f"Could not materialize or read linear model weights for {model_ref}: {e}")
        return []


def fetch_system_diagnostics(
    project_id: str,
    dataset_id: str,
    table_id: str,
    model_name: str = "restaurant_preference_model",
    linear_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch live model training metadata (for both primary Boosted Tree and
    companion Linear Regression model), table metadata, prediction drift,
    enrichment freshness ranges, once-per-model feature importance, and
    once-per-model linear weights."""
    resolved_linear_name = linear_model_name or (
        model_name if model_name.endswith("_linear") else f"{model_name}_linear"
    )
    diag: Dict[str, Any] = {
        "model_name": model_name,
        "model_trained_at": None,
        "vertex_version": None,
        "model_type": None,
        "feature_count": None,
        "mae": None,
        "r_squared": None,
        "iterations": None,
        "linear_model_name": resolved_linear_name,
        "linear_model_trained_at": None,
        "linear_model_type": None,
        "linear_feature_count": None,
        "linear_mae": None,
        "linear_r_squared": None,
        "table_modified_at": None,
        "table_total_rows": None,
        "feature_importance": [],
        "linear_weights": [],
    }
    try:
        client = bigquery.Client(project=project_id)
    except Exception as e:
        diag["error"] = str(e)
        return diag

    model_ref = f"{project_id}.{dataset_id}.{model_name}"
    linear_model_ref = f"{project_id}.{dataset_id}.{resolved_linear_name}"
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # 1a. Free BQML Primary Tree Model Metadata Read (0 bytes scanned)
    try:
        model = client.get_model(model_ref)
        diag["model_trained_at"] = getattr(model, "created", None)
        diag["model_type"] = getattr(model, "model_type", None)
        feature_cols = getattr(model, "feature_columns", None)
        if feature_cols is not None:
            diag["feature_count"] = len(feature_cols)

        props = getattr(model, "_properties", {}) or {}
        runs = props.get("trainingRuns") or getattr(model, "training_runs", None) or []
        if runs:
            latest_run = runs[-1] if isinstance(runs[-1], dict) else {}
            diag["vertex_version"] = latest_run.get("vertexAiModelVersion")
            reg_metrics = (
                (latest_run.get("evaluationMetrics") or {}).get("regressionMetrics") or {}
            )
            if "meanAbsoluteError" in reg_metrics:
                diag["mae"] = float(reg_metrics["meanAbsoluteError"])
            if "rSquared" in reg_metrics:
                diag["r_squared"] = float(reg_metrics["rSquared"])
            results = latest_run.get("results") or []
            if results:
                diag["iterations"] = len(results)
    except Exception as e:
        logger.warning(f"Could not read model metadata for {model_ref}: {e}")

    # 1b. Free BQML Companion Linear Model Metadata Read (0 bytes scanned)
    try:
        linear_model = client.get_model(linear_model_ref)
        linear_type = getattr(linear_model, "model_type", None)
        if linear_type in ("LINEAR_REGRESSION", "LINEAR_REG"):
            diag["linear_model_trained_at"] = getattr(linear_model, "created", None)
            diag["linear_model_type"] = linear_type
            lin_cols = getattr(linear_model, "feature_columns", None)
            if lin_cols is not None:
                diag["linear_feature_count"] = len(lin_cols)

            lin_props = getattr(linear_model, "_properties", {}) or {}
            lin_runs = (
                lin_props.get("trainingRuns")
                or getattr(linear_model, "training_runs", None)
                or []
            )
            if lin_runs:
                latest_lin_run = lin_runs[-1] if isinstance(lin_runs[-1], dict) else {}
                lin_reg_metrics = (
                    (latest_lin_run.get("evaluationMetrics") or {}).get("regressionMetrics")
                    or {}
                )
                if "meanAbsoluteError" in lin_reg_metrics:
                    diag["linear_mae"] = float(lin_reg_metrics["meanAbsoluteError"])
                if "rSquared" in lin_reg_metrics:
                    diag["linear_r_squared"] = float(lin_reg_metrics["rSquared"])
    except Exception as e:
        logger.warning(f"Could not read companion linear model metadata for {linear_model_ref}: {e}")

    # 2. Free Table Metadata Read (0 bytes scanned)
    try:
        table = client.get_table(table_ref)
        diag["table_modified_at"] = getattr(table, "modified", None)
        diag["table_total_rows"] = getattr(table, "num_rows", None)
    except Exception as e:
        logger.warning(f"Could not read table metadata for {table_ref}: {e}")

    # 3. 1-Row Drift & Freshness Summary Query (uses latest of tree & linear trained_at)
    tree_dt = _to_utc_dt(diag.get("model_trained_at"))
    linear_dt = _to_utc_dt(diag.get("linear_model_trained_at"))
    candidate_dts = [dt for dt in (tree_dt, linear_dt) if dt is not None]
    trained_dt = max(candidate_dts) if candidate_dts else None
    if trained_dt is not None:
        cutoff_expr = f"TIMESTAMP('{trained_dt.strftime('%Y-%m-%d %H:%M:%S+00:00')}')"
        current_pred_cond = f"predicted_at >= {cutoff_expr}"
        stale_pred_cond = f"(predicted_at IS NULL OR predicted_at < {cutoff_expr})"
    else:
        current_pred_cond = "predicted_at IS NOT NULL"
        stale_pred_cond = "FALSE"

    summary_sql = f"""
    SELECT
      COUNT(*) AS total_rows,
      COUNTIF(in_scope IS TRUE) AS in_scope_rows,
      COUNTIF(in_scope IS NULL) AS untriaged_rows,
      COUNTIF(in_scope IS TRUE AND user_rating IS NOT NULL) AS labeled_rows,
      CAST(MAX(first_seen) AS STRING) AS latest_first_seen,
      COUNTIF(in_scope IS TRUE AND predicted_user_rating IS NOT NULL AND {current_pred_cond}) AS current_predictions,
      COUNTIF(in_scope IS TRUE AND predicted_user_rating IS NOT NULL AND {stale_pred_cond}) AS stale_predictions,
      COUNTIF(in_scope IS TRUE AND predicted_user_rating IS NULL) AS unscored_in_scope,
      COUNTIF(in_scope IS TRUE AND gemini_insights_structured IS NOT NULL AND TRIM(gemini_insights_structured) != '') AS gemini_profiled_in_scope,
      MIN(IF(in_scope IS TRUE, gemini_profiled_at, NULL)) AS oldest_gemini_at,
      MAX(IF(in_scope IS TRUE, gemini_profiled_at, NULL)) AS newest_gemini_at,
      COUNTIF(in_scope IS TRUE AND maps_found IS NOT NULL) AS maps_checked_in_scope,
      MIN(IF(in_scope IS TRUE, maps_lookup_at, NULL)) AS oldest_maps_at,
      MAX(IF(in_scope IS TRUE, maps_lookup_at, NULL)) AS newest_maps_at
    FROM `{table_ref}`
    """
    try:
        rows = list(_wait_for_job(client.query(summary_sql), timeout=30.0))
        if rows:
            diag.update(dict(rows[0]))
    except Exception as e:
        logger.warning(f"Could not run diagnostics summary query on {table_ref}: {e}")

    # 4a. Primary Tree Model Feature Importance (materialized once per model_trained_at in BigQuery)
    if diag.get("model_trained_at") is not None:
        diag["feature_importance"] = ensure_model_feature_importance(
            project_id=project_id,
            dataset_id=dataset_id,
            model_name=model_name,
            model_trained_at=diag.get("model_trained_at"),
            vertex_version=diag.get("vertex_version"),
            client=client,
        )

    # 4b. Companion Linear Model Weights (materialized once per linear_model_trained_at in BigQuery)
    if diag.get("linear_model_trained_at") is not None:
        diag["linear_weights"] = ensure_model_linear_weights(
            project_id=project_id,
            dataset_id=dataset_id,
            linear_model_name=resolved_linear_name,
            model_trained_at=diag.get("linear_model_trained_at"),
            client=client,
        )

    return diag


MASTER_BQ_SCHEMA = [
    bigquery.SchemaField('fhrsid', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('businessname', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('addressline1', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('addressline2', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('addressline3', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('postcode', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('localauthorityname', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('ratingvalue', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('newratingpending', 'BOOLEAN', mode='NULLABLE'),
    bigquery.SchemaField('first_seen', 'DATE', mode='NULLABLE'),
    bigquery.SchemaField('user_rating', 'INT64', mode='NULLABLE'),
    bigquery.SchemaField('predicted_user_rating', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('tree_pred', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('lin_pred', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('predicted_at', 'TIMESTAMP', mode='NULLABLE'),
    bigquery.SchemaField('gemini_insights_structured', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('price_level', 'INT64', mode='NULLABLE'),
    bigquery.SchemaField('maps_rating', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('maps_reviews', 'INT64', mode='NULLABLE'),
    bigquery.SchemaField('latitude', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('longitude', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('maps_url', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('business_status', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('website_url', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('maps_types', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('in_scope', 'BOOLEAN', mode='NULLABLE'),
    bigquery.SchemaField('rating_source', 'STRING', mode='NULLABLE'),
]

# The pillar columns are appended from the canonical schema rather than retyped,
# so adding a field in `app/core/pillar_schema.py` cannot leave the cron's load
# schema behind. `append_to_bigquery` fills anything absent from the DataFrame
# with NA, so listing columns the weekly ingest never populates is harmless.
#
# This list must never name a column the live table lacks -- `load_table_from_json`
# fails the whole load if it does, and that load is the weekly ingest. Run
# `scripts/migrate_pillar_columns.py --execute` before extending PILLAR_FIELDS.
_PILLAR_BQ_TYPES = {'INT64': 'INT64', 'BOOL': 'BOOLEAN', 'STRING': 'STRING',
                    'TIMESTAMP': 'TIMESTAMP'}

MASTER_BQ_SCHEMA += [
    bigquery.SchemaField(name, _PILLAR_BQ_TYPES[bq_type], mode='NULLABLE')
    for name, bq_type in (
        [(f.column, f.bq_type) for f in PILLAR_FIELDS] + list(NON_JSON_COLUMNS)
    )
]
