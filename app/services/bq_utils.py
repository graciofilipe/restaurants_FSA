import pandas as pd
from google.cloud import bigquery, exceptions as google_cloud_exceptions
from google.cloud.bigquery.client import Client
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas_gbq
from google.auth.exceptions import DefaultCredentialsError
import logging
from scripts.bq_scripts import (
    SCRIPT_IDENTIFY_RECENTS, 
    SCRIPT_GENERATE_INSIGHTS, 
    SCRIPT_MERGE_INSIGHTS,
    SCRIPT_BULK_UPDATE_MERGE
)

# Configure logging
# In a library/service module, it's better to use getLogger and let the app configure the handlers
logger = logging.getLogger(__name__)

# Definition of columns to keep for processing new establishments
ORIGINAL_COLUMNS_TO_KEEP = [
    'FHRSID', 'BusinessName', 'AddressLine1', 'AddressLine2', 'AddressLine3',
    'PostCode', 'LocalAuthorityName', 'RatingValue', 'NewRatingPending',
    'first_seen', 'manual_review', 'gemini_insights', 'gemini_insights_structured'
]

# Custom Exceptions
class BigQueryExecutionError(Exception):
    """Custom exception for errors during BigQuery query execution."""
    pass

class DataFrameConversionError(Exception):
    """Custom exception for errors during DataFrame conversion from BigQuery results."""
    pass

# Module-level constant for FHRSID column name
FHRSID_COLNAME = "fhrsid"

def execute_gemini_enrichment(
    project_id: str,
    dataset_id: str,
    master_table_id: str,
    connection_id: str = 'eu.gemini',
    model_endpoint: str = 'gemini-3-pro-preview',
    days_recent: int = 33,
    review_status_filter: List[str] = None,
    excluded_locations: List[str] = None,
    fhrsids: List[str] = None
) -> bool:
    """
    Orchestrates the Gemini enrichment process using BigQuery scripts.
    If 'fhrsids' is provided, it processes only those IDs.
    Otherwise, it uses 'days_recent', 'review_status_filter', and 'excluded_locations'.
    """
    client = bigquery.Client(project=project_id)
    
    # Define intermediate table names
    recents_table_id = "recents"
    insights_table_id = "genairesults_temp"
    
    try:
        # Step 1: Identify Recents (or specific selection)
        logger.info("Step 1: Identifying target restaurants...")
        
        filter_condition = ""
        
        if fhrsids:
            # Explicit selection mode
            escaped_ids = [str(fid).replace("'", "''") for fid in fhrsids]
            id_list_str = ", ".join([f"'{fid}'" for fid in escaped_ids])
            # Use CAST to ensure compatibility if fhrsid is stored as INTEGER in BQ
            filter_condition = f"CAST(fhrsid AS STRING) IN ({id_list_str})"
            logger.info(f"Targeting {len(fhrsids)} specific FHRSIDs.")
        else:
            # Default filter mode
            if review_status_filter:
                status_list_str = ", ".join([f"'{s}'" for s in review_status_filter])
            else:
                status_list_str = "'pending', 'not reviewed'"
            
            exclusion_clause = ""
            if excluded_locations:
                escaped_locs = [loc.replace("'", "''") for loc in excluded_locations]
                locs_str = ", ".join([f"'{loc}'" for loc in escaped_locs])
                exclusion_clause = f"AND localauthorityname NOT IN ({locs_str})"
            
            filter_condition = f"DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {days_recent} AND manual_review IN ({status_list_str}) {exclusion_clause}"
            logger.info("Targeting recent restaurants based on date and status filters.")

        query_recents = SCRIPT_IDENTIFY_RECENTS.format(
            project_id=project_id,
            dataset_id=dataset_id,
            source_table=master_table_id,
            target_table_recents=recents_table_id,
            filter_condition=filter_condition
        )
        job1 = client.query(query_recents)
        job1.result()
        logger.info("Step 1 Complete.")

        # Step 2: Generate Insights
        logger.info("Step 2: Generating Gemini insights (this may take a while)...")
        query_insights = SCRIPT_GENERATE_INSIGHTS.format(
            project_id=project_id,
            dataset_id=dataset_id,
            source_table_recents=recents_table_id,
            target_table_insights=insights_table_id,
            connection_id=connection_id,
            model_endpoint=model_endpoint
        )
        job2 = client.query(query_insights)
        job2.result()
        logger.info("Step 2 Complete.")

        # Step 3: Merge Insights
        logger.info("Step 3: Merging insights back to master table...")
        query_merge = SCRIPT_MERGE_INSIGHTS.format(
            project_id=project_id,
            dataset_id=dataset_id,
            source_table_insights=insights_table_id,
            target_table_master=master_table_id
        )
        job3 = client.query(query_merge)
        job3.result()
        logger.info("Step 3 Complete.")
        
        return True

    except Exception as e:
        logger.error(f"Error during Gemini enrichment process: {e}")
        return False

def load_all_data_from_bq(project_id: str, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
    """
    Loads all data from a specified BigQuery table.
    """
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref_str}`"
    logger.info(f"Executing BigQuery query: {query}")

    try:
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        if df is not None and not df.empty:
            return df.to_dict(orient='records')
        else:
            return []
    except (pandas_gbq.gbq.GenericGBQException, DefaultCredentialsError) as e:
        logger.error(f"Error loading data from BigQuery table {table_ref_str}: {e}")
        return []
    except AttributeError as e:
        logger.error(f"AttributeError during DataFrame processing for {table_ref_str}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data from BigQuery table {table_ref_str}: {e}")
        return []

def load_filtered_data_from_bq(
    project_id: str,
    dataset_id: str,
    table_id: str,
    days_filter: int = None,
    review_status_filter: List[str] = None,
    excluded_locations: List[str] = None,
    postcode_areas: List[str] = None,
    gemini_insights_status: str = None
) -> List[Dict[str, Any]]:
    """
    Loads data from BigQuery with optional filters.
    """
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    query = f"SELECT * FROM `{table_ref_str}` WHERE 1=1"
    
    if days_filter is not None:
        query += f" AND DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {days_filter}"
    
    if review_status_filter:
        statuses_str = ", ".join([f"'{s}'" for s in review_status_filter])
        query += f" AND manual_review IN ({statuses_str})"
    
    if excluded_locations:
        # Escape single quotes in location names just in case
        escaped_locs = [loc.replace("'", "''") for loc in excluded_locations]
        locs_str = ", ".join([f"'{loc}'" for loc in escaped_locs])
        query += f" AND localauthorityname NOT IN ({locs_str})"
    
    if postcode_areas:
        escaped_pcs = [pc.replace("'", "''") for pc in postcode_areas]
        pcs_str = ", ".join([f"'{pc}'" for pc in escaped_pcs])
        query += f" AND SPLIT(postcode, ' ')[SAFE_OFFSET(0)] IN ({pcs_str})"

    if gemini_insights_status:
        if gemini_insights_status.lower() == 'populated':
            query += " AND gemini_insights IS NOT NULL"
        elif gemini_insights_status.lower() == 'null':
            query += " AND gemini_insights IS NULL"
        
    logger.info(f"Executing Filtered BigQuery query: {query}")

    try:
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        if df is not None and not df.empty:
            if 'first_seen' in df.columns:
                 df['first_seen'] = df['first_seen'].astype(str)
            return df.to_dict(orient='records')
        else:
            return []
    except Exception as e:
        logger.error(f"Error loading filtered data from {table_ref_str}: {e}")
        return []

def sanitize_column_name(column_name: str) -> str:
    """
    Sanitizes a column name for BigQuery compatibility.
    """
    name = column_name.replace(' ', '_')
    name = name.replace('.', '')
    name = name.replace('@', '')
    name = name.replace('-', '_')
    
    name = name.lower()
    
    if name and not name[0].isalnum() and name[0] != '_':
        name = name[1:]

    name = re.sub(r'[^a-z0-9_]+', '_', name)
    name = name.strip('_')
    
    if not name:
        return "unnamed_column"
        
    return name

def bulk_update_reviews(
    project_id: str,
    dataset_id: str,
    target_table_id: str,
    df_updates: pd.DataFrame
) -> Tuple[bool, str]:
    """
    Performs a bulk update of the 'manual_review' column using a temporary table and MERGE.
    """
    if df_updates.empty:
        logger.warning("DataFrame for bulk update is empty.")
        return False, "DataFrame is empty."

    df_updates = df_updates.copy()
    df_updates.columns = [col.lower() for col in df_updates.columns]

    logger.debug(f"df_updates normalized columns: {df_updates.columns.tolist()}")
    required_cols = ['fhrsid', 'manual_review']
    if not all(col in df_updates.columns for col in required_cols):
        logger.error(f"DataFrame missing required columns: {required_cols}. Found: {df_updates.columns.tolist()}")
        return False, f"Missing columns. Required: {required_cols}, Found: {df_updates.columns.tolist()}"

    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    temp_table_id = f"temp_update_reviews_{timestamp_str}"
    
    logger.info(f"Initiating bulk update using temp table: {temp_table_id}")

    temp_schema = [
        bigquery.SchemaField("fhrsid", "STRING"),
        bigquery.SchemaField("manual_review", "STRING")
    ]
    
    success_upload = write_to_bigquery(
        df=df_updates,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=temp_table_id,
        columns_to_select=required_cols,
        bq_schema=temp_schema
    )

    if not success_upload:
        logger.error(f"Failed to upload temporary update table {temp_table_id}. Aborting bulk update.")
        return False, "Failed to upload temporary table to BigQuery."

    client = bigquery.Client(project=project_id)
    
    try:
        query = SCRIPT_BULK_UPDATE_MERGE.format(
            project_id=project_id,
            dataset_id=dataset_id,
            target_table=target_table_id,
            source_table_temp=temp_table_id
        )
        
        logger.debug(f"Executing MERGE query:\n{query}")
        query_job = client.query(query)
        query_job.result()
        affected_rows = query_job.num_dml_affected_rows
        logger.info(f"MERGE query completed. Rows affected: {affected_rows}")
        
        table_ref_str = f"{project_id}.{dataset_id}.{temp_table_id}"
        client.delete_table(table_ref_str, not_found_ok=True)
        logger.info(f"Temporary table {table_ref_str} deleted.")
        
        return True, f"{affected_rows} rows updated."

    except Exception as e:
        logger.error(f"Error during bulk update execution: {e}")
        return False, f"Error executing update: {str(e)}"


def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, columns_to_select: List[str], bq_schema: List[bigquery.SchemaField]) -> bool:
    """
    Writes a Pandas DataFrame to a BigQuery table.
    """
    df_subset = df[columns_to_select].copy()

    if 'Geocode.Latitude' in df_subset.columns:
        df_subset['Geocode.Latitude'] = pd.to_numeric(df_subset['Geocode.Latitude'], errors='coerce')
    if 'Geocode.Longitude' in df_subset.columns:
        df_subset['Geocode.Longitude'] = pd.to_numeric(df_subset['Geocode.Longitude'], errors='coerce')

    original_columns = df_subset.columns.tolist()
    sanitized_columns = [sanitize_column_name(col) for col in original_columns]
    df_subset.columns = sanitized_columns
    
    original_new_rating_pending_col = 'NewRatingPending'
    sanitized_new_rating_pending_col = sanitize_column_name(original_new_rating_pending_col)

    if sanitized_new_rating_pending_col in df_subset.columns:
        logger.debug(f"Unique values in {sanitized_new_rating_pending_col} before conversion: {df_subset[sanitized_new_rating_pending_col].unique()}")
        mapping = {'true': True, 'false': False}
        df_subset[sanitized_new_rating_pending_col] = df_subset[sanitized_new_rating_pending_col].astype(str).str.lower().map(mapping).fillna(pd.NA)
    else:
        logger.warning(f"Column {sanitized_new_rating_pending_col} not found in df_subset.")

    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        column_name_character_map="V2",
    )

    # Detailed logging for debugging
    # logger.debug(f"BigQuery job_config.schema: {job_config.schema}")
    
    sanitized_fhrsid_col = 'fhrsid'
    if sanitized_fhrsid_col in df_subset.columns:
        if df_subset[sanitized_fhrsid_col].dtype != 'object':
            df_subset[sanitized_fhrsid_col] = df_subset[sanitized_fhrsid_col].astype(str)
    else:
        logger.warning(f"Column '{sanitized_fhrsid_col}' not found in DataFrame during write operation.")

    try:
        job = client.load_table_from_dataframe(df_subset, table_ref_str, job_config=job_config)
        job.result()
        logger.info(f"Successfully wrote data to BigQuery table {table_ref_str}.")
        return True
    except Exception as e:
        logger.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False

def append_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, bq_schema: List[bigquery.SchemaField]) -> bool:
    """
    Appends a Pandas DataFrame to an existing BigQuery table.
    """
    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

    schema_columns = [field.name for field in bq_schema]
    df_subset = df[schema_columns].copy()

    geocode_latitude_col = 'geocode_latitude'
    geocode_longitude_col = 'geocode_longitude'
    new_rating_pending_col = 'newratingpending'

    if geocode_latitude_col in df_subset.columns:
        df_subset[geocode_latitude_col] = pd.to_numeric(df_subset[geocode_latitude_col], errors='coerce')
    if geocode_longitude_col in df_subset.columns:
        df_subset[geocode_longitude_col] = pd.to_numeric(df_subset[geocode_longitude_col], errors='coerce')

    if new_rating_pending_col in df_subset.columns:
        mapping = {'true': True, 'false': False, 'TRUE': True, 'FALSE': False}
        df_subset[new_rating_pending_col] = df_subset[new_rating_pending_col].astype(str).str.lower().map(mapping)
        df_subset[new_rating_pending_col] = df_subset[new_rating_pending_col].astype('boolean')

    if 'first_seen' in df_subset.columns:
        # Convert to datetime then date object for BQ compatibility
        df_subset['first_seen'] = pd.to_datetime(df_subset['first_seen'], errors='coerce').dt.date

    fhrsid_col_name = 'fhrsid'
    if fhrsid_col_name in df_subset.columns:
        fhrsid_bq_type = None
        for field in bq_schema:
            if field.name == fhrsid_col_name:
                fhrsid_bq_type = field.field_type
                break

        if fhrsid_bq_type:
            if fhrsid_bq_type in ['INTEGER', 'INT64', 'NUMERIC']:
                df_subset[fhrsid_col_name] = pd.to_numeric(df_subset[fhrsid_col_name], errors='coerce')
            elif fhrsid_bq_type == 'STRING':
                df_subset[fhrsid_col_name] = df_subset[fhrsid_col_name].astype(str)
        else:
            logger.warning(f"Column '{fhrsid_col_name}' (for FHRSID) not found in bq_schema.")
    else:
        logger.warning(f"Column '{fhrsid_col_name}' (for FHRSID) not found in DataFrame for append_to_bigquery.")

    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        column_name_character_map="V2",
    )

    try:
        job = client.load_table_from_dataframe(df_subset, table_ref_str, job_config=job_config)
        job.result()  # Wait for the job to complete
        logger.info(f"Successfully appended data to BigQuery table {table_ref_str}.")
        return True
    except Exception as e:
        logger.error(f"Error appending data to BigQuery table {table_ref_str}: {e}")
        return False

def update_rows_in_bigquery(project_id: str, dataset_id: str, table_id: str, fhrsid: str, update_data: Dict[str, Any]) -> bool:
    """
    Updates specific rows in a BigQuery table based on FHRSID.
    """
    if not update_data:
        logger.warning("No data provided for update.")
        return False

    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

    set_clauses = []
    for column, value in update_data.items():
        if isinstance(value, str):
            sanitized_value = value.replace("'", "''").replace('\\', '\\\\')
            set_clauses.append(f"`{column}` = '{sanitized_value}'")
        elif isinstance(value, bool):
            set_clauses.append(f"`{column}` = {str(value).upper()}")
        elif isinstance(value, (int, float)):
            set_clauses.append(f"`{column}` = {value}")
        elif value is None:
            set_clauses.append(f"`{column}` = NULL")
        else:
            sanitized_value = str(value).replace("'", "''").replace('\\', '\\\\')
            logger.warning(f"Column '{column}' has an unhandled type {type(value)}. Converting to string: '{sanitized_value}'")
            set_clauses.append(f"`{column}` = '{sanitized_value}'")


    if not set_clauses:
        logger.warning("No valid SET clauses generated from update_data.")
        return False

    set_statement = ", ".join(set_clauses)
    escaped_fhrsid_value = fhrsid.replace("'", "''")
    query = f"UPDATE `{table_ref_str}` SET {set_statement} WHERE {FHRSID_COLNAME} = '{escaped_fhrsid_value}'"

    logger.info(f"Executing BigQuery UPDATE query: {query}")

    try:
        query_job = client.query(query)
        query_job.result()
        if query_job.errors:
            logger.error(f"BigQuery UPDATE failed with errors: {query_job.errors}")
            return False
        logger.info(f"Successfully updated rows in {table_ref_str} for {FHRSID_COLNAME} = '{escaped_fhrsid_value}'.")
        return True
    except DefaultCredentialsError as e:
        logger.error(f"BigQuery authentication error: {e}. Ensure your environment is configured correctly for ADC.")
        return False
    except Exception as e:
        logger.error(f"An error occurred during BigQuery UPDATE: {e}")
        return False

def execute_merge_query(merge_query: str, project_id: str) -> bool:
    """
    Executes a MERGE SQL query in BigQuery.
    """
    logger.info(f"Attempting to execute MERGE query in project '{project_id}':\n{merge_query}")
    try:
        client = bigquery.Client(project=project_id)
        query_job = client.query(merge_query)
        query_job.result()

        if query_job.errors:
            logger.error(f"MERGE query failed with errors: {query_job.errors}")
            return False

        logger.info("MERGE query executed successfully.")
        return True
    except DefaultCredentialsError as e:
        logger.error(f"BigQuery authentication error during MERGE query execution: {e}. Ensure ADC is configured.")
        return False
    except google_cloud_exceptions.GoogleCloudError as e:
        logger.error(f"A Google Cloud error occurred during MERGE query execution: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during MERGE query execution: {e}")
        return False

def get_distinct_local_authorities(project_id: str, dataset_id: str, table_id: str) -> List[str]:
    """
    Fetches a list of distinct LocalAuthorityName values from the master table.
    """
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT DISTINCT localauthorityname FROM `{table_ref}` WHERE localauthorityname IS NOT NULL ORDER BY localauthorityname"
    
    logger.info(f"Fetching distinct Local Authorities from {table_ref}")
    try:
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        if df is not None and not df.empty:
            count = len(df)
            logger.info(f"Fetched {count} distinct Local Authorities from {table_ref}.")
            return df['localauthorityname'].tolist()
        return []
    except Exception as e:
        logger.error(f"Error fetching distinct local authorities: {e}")
        return []

def get_distinct_outcodes(project_id: str, dataset_id: str, table_id: str) -> List[str]:
    """
    Fetches a list of distinct Postcode Areas (outcodes) from the master table.
    """
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"""
        SELECT DISTINCT SPLIT(postcode, ' ')[SAFE_OFFSET(0)] as outcode 
        FROM `{table_ref}` 
        WHERE postcode IS NOT NULL 
        ORDER BY outcode
    """
    
    logger.info(f"Fetching distinct Outcodes from {table_ref}")
    try:
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        if df is not None and not df.empty:
            outcodes = df['outcode'].dropna().astype(str).tolist()
            return sorted([o for o in outcodes if o.strip()])
        return []
    except Exception as e:
        logger.error(f"Error fetching distinct outcodes: {e}")
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
    bigquery.SchemaField('gemini_insights', 'STRING', mode='NULLABLE'),
    bigquery.SchemaField('gemini_insights_structured', 'STRING', mode='NULLABLE'),
]

def upsert_agent_insight(project_id: str, dataset_id: str, table_id: str, insight_data: Dict[str, Any]) -> bool:
    """
    Upserts agent insights into the specified BigQuery table.
    """
    if not insight_data or 'fhrsid' not in insight_data:
        logger.error("Invalid insight data provided for upsert.")
        return False
        
    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    # Construct MERGE query
    query = f"""
    MERGE `{table_ref_str}` T
    USING (
        SELECT 
            @fhrsid as fhrsid,
            @raw_insight as raw_insight,
            @cuisine_type as cuisine_type,
            @review_count as review_count,
            @average_rating as average_rating,
            @updated_at as updated_at
    ) S
    ON T.fhrsid = S.fhrsid
    WHEN MATCHED THEN
      UPDATE SET 
        raw_insight = S.raw_insight,
        cuisine_type = S.cuisine_type,
        review_count = S.review_count,
        average_rating = S.average_rating,
        updated_at = S.updated_at
    WHEN NOT MATCHED THEN
      INSERT (fhrsid, raw_insight, cuisine_type, review_count, average_rating, updated_at)
      VALUES (S.fhrsid, S.raw_insight, S.cuisine_type, S.review_count, S.average_rating, S.updated_at)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("fhrsid", "STRING", str(insight_data.get("fhrsid"))),
            bigquery.ScalarQueryParameter("raw_insight", "STRING", insight_data.get("raw_insight")),
            bigquery.ScalarQueryParameter("cuisine_type", "STRING", insight_data.get("cuisine_type")),
            bigquery.ScalarQueryParameter("review_count", "INT64", insight_data.get("review_count")),
            bigquery.ScalarQueryParameter("average_rating", "FLOAT64", insight_data.get("average_rating")),
            bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", insight_data.get("updated_at")),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        query_job.result()
        logger.info(f"Successfully upserted agent insight for FHRSID {insight_data.get('fhrsid')}.")
        return True
    except Exception as e:
        logger.error(f"Error upserting agent insight: {e}")
        return False

def load_specific_agent_insights(project_id: str, dataset_id: str, fhrsids: List[str]) -> List[Dict[str, Any]]:
    """
    Loads agent insights from BigQuery for a specific list of FHRSIDs.
    """
    if not fhrsids:
        return []
        
    client = bigquery.Client(project=project_id)
    table_id = "restaurant_agent_insights"
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    query = f"""
        SELECT *
        FROM `{table_ref_str}`
        WHERE fhrsid IN UNNEST(@fhrsids)
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("fhrsids", "STRING", [str(fid) for fid in fhrsids])
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        records = []
        for row in results:
            # results object allows dictionary-like access
            record = {key: value for key, value in row.items()}
            # Convert timestamp to string for display/session compatibility
            if 'updated_at' in record and hasattr(record['updated_at'], 'isoformat'):
                record['updated_at'] = record['updated_at'].isoformat()
            records.append(record)
            
        return records
    except Exception as e:
        logger.error(f"Error loading specific agent insights: {e}")
        return []