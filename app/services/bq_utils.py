import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery, exceptions as google_cloud_exceptions
import pandas as pd
from app.core.pillar_schema import sql_conformance_check, summarise_conformance
from scripts.bq_scripts import (
    MODEL_PARAMS_JSON,
    SCRIPT_BULK_UPDATE_MERGE,
    SCRIPT_GENERATE_INSIGHTS,
    SCRIPT_IDENTIFY_RECENTS,
    SCRIPT_MERGE_INSIGHTS,
)

logger = logging.getLogger(__name__)

ORIGINAL_COLUMNS_TO_KEEP = [
    'FHRSID', 'BusinessName', 'AddressLine1', 'AddressLine2', 'AddressLine3',
    'PostCode', 'LocalAuthorityName', 'RatingValue', 'NewRatingPending',
    'first_seen', 'manual_review', 'gemini_insights', 'gemini_insights_structured'
]

class BigQueryExecutionError(Exception):
    pass

class DataFrameConversionError(Exception):
    pass

FHRSID_COLNAME = "fhrsid"

def _sql_quote(val: Any) -> str:
    s = str(val).replace("'", "''")
    return f"'{s}'"

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
        rows = list(client.query(sql_conformance_check(table_ref)).result())
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
    review_status_filter: Optional[List[str]] = None,
    excluded_locations: Optional[List[str]] = None,
    fhrsids: Optional[List[str]] = None,
) -> bool:
    """Orchestrates the Gemini enrichment process using BigQuery SQL scripts."""
    client = bigquery.Client(project=project_id)
    # Per-run names: these scratch tables sit in the production dataset, and two
    # overlapping runs sharing `recents` would each profile the other's selection.
    run_id = uuid.uuid4().hex[:12]
    recents_table_id, insights_table_id = f"recents_{run_id}", f"genairesults_temp_{run_id}"
    try:
        if fhrsids:
            escaped = [_sql_quote(f) for f in fhrsids]
            filter_condition = f"CAST(fhrsid AS STRING) IN ({', '.join(escaped)})"
        else:
            status_str = ", ".join(_sql_quote(s) for s in (review_status_filter or ['pending', 'not reviewed']))
            excl_clause = ""
            if excluded_locations:
                escaped_locs = [_sql_quote(l) for l in excluded_locations]
                excl_clause = f"AND localauthorityname NOT IN ({', '.join(escaped_locs)})"
            filter_condition = f"DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {days_recent} AND manual_review IN ({status_str}) {excl_clause}"

        q_recents = SCRIPT_IDENTIFY_RECENTS.format(
            project_id=project_id, dataset_id=dataset_id, source_table=master_table_id,
            target_table_recents=recents_table_id, filter_condition=filter_condition
        )
        client.query(q_recents).result()

        q_insights = SCRIPT_GENERATE_INSIGHTS.format(
            project_id=project_id, dataset_id=dataset_id, source_table_recents=recents_table_id,
            target_table_insights=insights_table_id, connection_id=connection_id,
            model_endpoint=model_endpoint, model_params_json=MODEL_PARAMS_JSON
        )
        client.query(q_insights).result()

        log_insight_conformance(client, project_id, dataset_id, insights_table_id)

        q_merge = SCRIPT_MERGE_INSIGHTS.format(
            project_id=project_id, dataset_id=dataset_id, source_table_insights=insights_table_id,
            target_table_master=master_table_id
        )
        job = client.query(q_merge)
        job.result()
        return True
    except Exception as e:
        logger.error(f"Error during Gemini enrichment: {e}")
        return False
    finally:
        for temp_table_id in (recents_table_id, insights_table_id):
            try:
                client.delete_table(f"{project_id}.{dataset_id}.{temp_table_id}", not_found_ok=True)
            except Exception as e:
                # The table expires on its own; a failed drop is not worth failing the run.
                logger.warning(f"Could not drop temp table {temp_table_id}: {e}")

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
    review_status_filter: Optional[List[str]] = None,
    excluded_locations: Optional[List[str]] = None,
    postcode_areas: Optional[List[str]] = None,
    gemini_insights_status: Optional[str] = None,
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
    if review_status_filter:
        escaped_statuses = [_sql_quote(s) for s in review_status_filter]
        query += f" AND manual_review IN ({', '.join(escaped_statuses)})"
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
    if gemini_insights_status:
        query += " AND gemini_insights IS NOT NULL" if gemini_insights_status.lower() == 'populated' else " AND gemini_insights IS NULL"

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
        return []

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
    """Performs a bulk update of manual_review, in_scope, user_rating, and/or rating_source columns using a temp table and MERGE."""
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

    possible_update_cols = ['manual_review', 'user_rating', 'in_scope', 'rating_source']
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
        elif col in ['manual_review', 'rating_source']:
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

    for geo in ['Geocode.Latitude', 'Geocode.Longitude']:
        if geo in df_sub.columns:
            df_sub[geo] = pd.to_numeric(df_sub[geo], errors='coerce')

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

    for geo in ['geocode_latitude', 'geocode_longitude']:
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
        return []

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
        return []

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
    bigquery.SchemaField('manual_review', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('user_rating', 'INT64', mode='NULLABLE'),
    bigquery.SchemaField('predicted_user_rating', 'FLOAT64', mode='NULLABLE'),
    bigquery.SchemaField('predicted_at', 'TIMESTAMP', mode='NULLABLE'),
    bigquery.SchemaField('gemini_insights', 'STRING', mode='NULLABLE'),
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
