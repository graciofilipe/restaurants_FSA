import streamlit as st
import pandas as pd

# Definition of columns to keep for processing new establishments
ORIGINAL_COLUMNS_TO_KEEP = [
    'FHRSID', 'BusinessName', 'AddressLine1', 'AddressLine2', 'AddressLine3',
    'PostCode', 'LocalAuthorityName', 'RatingValue', 'NewRatingPending',
    'first_seen', 'manual_review', 'gemini_insights'
]

from google.cloud import bigquery
from google.cloud.bigquery.client import Client # Explicit import for type hinting if needed elsewhere
from typing import List, Dict, Any # Removed Optional, Added Dict, Any
import re
import pandas_gbq # Added import
from google.auth.exceptions import DefaultCredentialsError # Added import

# Custom Exceptions
class BigQueryExecutionError(Exception):
    """Custom exception for errors during BigQuery query execution."""
    pass

class DataFrameConversionError(Exception):
    """Custom exception for errors during DataFrame conversion from BigQuery results."""
    pass

# Module-level constant for FHRSID column name
FHRSID_COLNAME = "fhrsid"

def load_all_data_from_bq(project_id: str, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
    """
    Loads all data from a specified BigQuery table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.

    Returns:
        A list of dictionaries, where each dictionary represents a row from the table.
        Returns an empty list if the table is empty or if an error occurs during the process.
    """
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref_str}`"
    print(f"Executing BigQuery query: {query}")

    try:
        df = pandas_gbq.read_gbq(query, project_id=project_id)
        if df is not None and not df.empty:
            return df.to_dict(orient='records')
        else:
            return []
    except (pandas_gbq.gbq.GenericGBQException, DefaultCredentialsError) as e:
        print(f"Error loading data from BigQuery table {table_ref_str}: {e}")
        # As per plan, return an empty list in case of failure.
        # If re-raising was preferred: raise BigQueryExecutionError(f"Failed to load data from {table_ref_str}") from e
        return []
    except AttributeError as e: # Handles case where df might be None and .empty or .to_dict is called
        print(f"AttributeError during DataFrame processing for {table_ref_str}: {e}. This might indicate an issue with read_gbq's return value.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading data from BigQuery table {table_ref_str}: {e}")
        # As per plan, return an empty list for other unexpected errors.
        # If re-raising was preferred: raise BigQueryExecutionError(f"An unexpected error occurred while loading data from {table_ref_str}") from e
        return []

def get_recent_restaurants(N_DAYS, project_id: str, dataset_id: str, table_id: str):

    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

    query = f'SELECT * FROM {table_ref_str}  WHERE \
          DATE_DIFF(CURRENT_DATE(), first_seen, DAY) < {N_DAYS}'

    new_df = pd.read_gbq(query, project_id=project_id)

    return new_df


def sanitize_column_name(column_name: str) -> str:
    """
    Sanitizes a column name for BigQuery compatibility.
    - Removes spaces, periods, '@' signs, and dashes.
    - Converts to lowercase.
    - Replaces sequences of non-alphanumeric characters (excluding underscores) with a single underscore.
    - Ensures the name doesn't start or end with an underscore.
    - Handles potential leading characters like '?' or '@' from json_normalize.
    """
    # Replace problematic characters with underscore or remove them
    # Order matters: handle specific removals before general non-alphanumeric replacement
    name = column_name.replace(' ', '_')
    name = name.replace('.', '')  # Remove periods
    name = name.replace('@', '')  # Remove @
    name = name.replace('-', '_') # Replace dash with underscore
    
    name = name.lower()
    
    # Remove any leading characters that are not alphanumeric or underscore
    # This helps with characters like '?' often added by json_normalize
    if name and not name[0].isalnum() and name[0] != '_':
        name = name[1:]

    # Replace any remaining sequence of non-alphanumeric characters (except underscore) with a single underscore
    name = re.sub(r'[^a-z0-9_]+', '_', name)
    
    # Ensure it doesn't start or end with an underscore
    name = name.strip('_')
    
    # If the name becomes empty after stripping (e.g. was "___"), provide a default
    if not name:
        return "unnamed_column"
        
    return name

def write_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, columns_to_select: List[str], bq_schema: List[bigquery.SchemaField]) -> bool:
    """
    Writes a Pandas DataFrame to a BigQuery table, with column selection and schema definition.

    Args:
        df: The DataFrame to write.
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        columns_to_select: A list of column names to select from the DataFrame.
        bq_schema: A list of bigquery.SchemaField objects for the BigQuery table.

    Returns:
        True if the write operation was successful, False otherwise.
    """
    # Subset the DataFrame to include only the selected columns
    df_subset = df[columns_to_select].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Convert Geocode columns to numeric, coercing errors to NaN
    if 'Geocode.Latitude' in df_subset.columns:
        df_subset['Geocode.Latitude'] = pd.to_numeric(df_subset['Geocode.Latitude'], errors='coerce')
    if 'Geocode.Longitude' in df_subset.columns:
        df_subset['Geocode.Longitude'] = pd.to_numeric(df_subset['Geocode.Longitude'], errors='coerce')

    # Sanitize column names for the subset
    original_columns = df_subset.columns.tolist()
    sanitized_columns = [sanitize_column_name(col) for col in original_columns]
    
    df_subset.columns = sanitized_columns
    
    # --- BEGIN MODIFICATIONS ---

    # Determine the sanitized column name for 'NewRatingPending'
    # Assuming 'NewRatingPending' is the original name and sanitize_column_name handles it correctly.
    original_new_rating_pending_col = 'NewRatingPending' # Original name before sanitization
    sanitized_new_rating_pending_col = sanitize_column_name(original_new_rating_pending_col)

    # Log unique values before conversion if the column exists
    if sanitized_new_rating_pending_col in df_subset.columns:
        print(f"Unique values in {sanitized_new_rating_pending_col} before conversion: {df_subset[sanitized_new_rating_pending_col].unique()}")

        # Convert 'NewRatingPending' (sanitized version) to Boolean
        # Define the mapping for string to boolean
        mapping = {
            'true': True,
            'false': False
        }

        # Apply the mapping
        # Ensure that the column is treated as string type before applying .str.lower()
        df_subset[sanitized_new_rating_pending_col] = df_subset[sanitized_new_rating_pending_col].astype(str).str.lower().map(mapping).fillna(pd.NA)
    else:
        print(f"Column {sanitized_new_rating_pending_col} (sanitized from {original_new_rating_pending_col}) not found in df_subset.columns: {df_subset.columns}")


    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        column_name_character_map="V2",
    )

    # Detailed Logging before BQ load
    print(f"BigQuery job_config.schema: {job_config.schema}")
    print(f"DataFrame dtypes before BQ load: \n{df_subset.dtypes}")
    print(f"Sample data for BQ load (first 5 rows): \n{df_subset.head().to_string()}")
    if sanitized_new_rating_pending_col in df_subset.columns:
        print(f"Unique values in {sanitized_new_rating_pending_col} after conversion: {df_subset[sanitized_new_rating_pending_col].unique()}")
        print(f"Data type of {sanitized_new_rating_pending_col} after conversion: {df_subset[sanitized_new_rating_pending_col].dtype}")
    
    # --- END MODIFICATIONS ---

    # Ensure fhrsid is string.
    sanitized_fhrsid_col = 'fhrsid' # Based on sanitize_column_name('fhrsid')
    if sanitized_fhrsid_col in df_subset.columns:
        if df_subset[sanitized_fhrsid_col].dtype != 'object': # Check if it's not a string type
            df_subset[sanitized_fhrsid_col] = df_subset[sanitized_fhrsid_col].astype(str)
    else:
        # This case should ideally not happen if fhrsid is expected
        print(f"Warning: Column '{sanitized_fhrsid_col}' not found in DataFrame during write operation.")

    try:
        job = client.load_table_from_dataframe(df_subset, table_ref_str, job_config=job_config)
        job.result()
        st.success(f"Successfully wrote data to BigQuery table {table_ref_str} with schema and sanitized column names. Overwritten if table existed.")
        return True
    except Exception as e:
        st.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False

def append_to_bigquery(df: pd.DataFrame, project_id: str, dataset_id: str, table_id: str, bq_schema: List[bigquery.SchemaField]) -> bool:
    """
    Appends a Pandas DataFrame to an existing BigQuery table.

    Args:
        df: The DataFrame to append. Assumes column names are already sanitized.
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        bq_schema: A list of bigquery.SchemaField objects for the BigQuery table.

    Returns:
        True if the append operation was successful, False otherwise.
    """
    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

    # Assume df columns are already sanitized as per requirements.
    # Select only columns defined in bq_schema
    schema_columns = [field.name for field in bq_schema]
    df_subset = df[schema_columns].copy()

    # Convert specific columns, assuming sanitized names
    # Example sanitized names used here, adjust if actual sanitization differs
    geocode_latitude_col = 'geocode_latitude' # Sanitized from 'Geocode.Latitude'
    geocode_longitude_col = 'geocode_longitude' # Sanitized from 'Geocode.Longitude'
    new_rating_pending_col = 'newratingpending' # Sanitized from 'NewRatingPending'

    if geocode_latitude_col in df_subset.columns:
        df_subset[geocode_latitude_col] = pd.to_numeric(df_subset[geocode_latitude_col], errors='coerce')
    if geocode_longitude_col in df_subset.columns:
        df_subset[geocode_longitude_col] = pd.to_numeric(df_subset[geocode_longitude_col], errors='coerce')

    if new_rating_pending_col in df_subset.columns:
        # Convert 'NewRatingPending' (sanitized version) to Boolean
        mapping = {'true': True, 'false': False, 'TRUE': True, 'FALSE': False}
        # Ensure column is string before .lower() or .map()
        df_subset[new_rating_pending_col] = df_subset[new_rating_pending_col].astype(str).str.lower().map(mapping)
        # Convert to pandas Boolean type to handle NA properly if needed, though BQ might handle True/False/None directly
        df_subset[new_rating_pending_col] = df_subset[new_rating_pending_col].astype('boolean')

    # Dynamically handle fhrsid data type based on BigQuery schema for append operations.
    # Record of previous attempts/fixes:
    # - Originally, fhrsid was often cast to string by default in write/append functions.
    # - This caused "Could not convert 'value' with type str: tried to convert to int64"
    #   errors when appending to tables like 'fsa_master' where 'fhrsid' is INT64.
    # - This fix inspects the bq_schema to apply appropriate type conversion for 'fhrsid'.
    fhrsid_col_name = 'fhrsid' # Assuming 'fhrsid' is the sanitized column name in bq_schema and df_subset

    if fhrsid_col_name in df_subset.columns:
        fhrsid_bq_type = None
        for field in bq_schema:
            if field.name == fhrsid_col_name:
                fhrsid_bq_type = field.field_type
                break

        if fhrsid_bq_type:
            current_type = df_subset[fhrsid_col_name].dtype
            print(f"FHRSID_DEBUG: Column '{fhrsid_col_name}' current initial dtype: {current_type}, Target BQ schema type: {fhrsid_bq_type}")

            if fhrsid_bq_type in ['INTEGER', 'INT64', 'NUMERIC']:
                # Convert to numeric. pd.to_numeric handles various input types including strings.
                # errors='coerce' will turn unparseable strings into NaN.
                print(f"FHRSID_DEBUG: Converting column '{fhrsid_col_name}' to numeric for BQ type {fhrsid_bq_type}.")
                df_subset[fhrsid_col_name] = pd.to_numeric(df_subset[fhrsid_col_name], errors='coerce')
                # Note: If FHRSID is a primary key or non-nullable INT64, NaNs (from coercion errors) could be an issue.
                # This matches behavior of Geocode coordinate coercion.
                print(f"FHRSID_DEBUG: Column '{fhrsid_col_name}' dtype after pd.to_numeric: {df_subset[fhrsid_col_name].dtype}")

            elif fhrsid_bq_type == 'STRING':
                # Always convert to string if BQ schema type is STRING to ensure all elements are strings.
                print(f"FHRSID_DEBUG: Converting column '{fhrsid_col_name}' to string for BQ type {fhrsid_bq_type}. Current dtype: {current_type}")
                df_subset[fhrsid_col_name] = df_subset[fhrsid_col_name].astype(str)
                print(f"FHRSID_DEBUG: Column '{fhrsid_col_name}' dtype after .astype(str): {df_subset[fhrsid_col_name].dtype}")
            else:
                print(f"FHRSID_DEBUG: Column '{fhrsid_col_name}' is type {fhrsid_bq_type} in BQ schema. No explicit fhrsid-specific conversion applied here.")
        else:
            print(f"Warning: Column '{fhrsid_col_name}' (for FHRSID) not found in bq_schema. No fhrsid-specific type conversion applied.")
    else:
        print(f"Warning: Column '{fhrsid_col_name}' (for FHRSID) not found in DataFrame for append_to_bigquery.")

    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        column_name_character_map="V2", # As in write_to_bigquery
    )

    try:
        job = client.load_table_from_dataframe(df_subset, table_ref_str, job_config=job_config)
        job.result()  # Wait for the job to complete
        st.success(f"Successfully appended data to BigQuery table {table_ref_str}.")
        return True
    except Exception as e:
        st.error(f"Error appending data to BigQuery table {table_ref_str}: {e}")
        # Also print to console for backend logging
        print(f"Error appending data to BigQuery table {table_ref_str}: {e}")
        return False

def update_rows_in_bigquery(project_id: str, dataset_id: str, table_id: str, fhrsid: str, update_data: Dict[str, Any]) -> bool:
    """
    Updates specific rows in a BigQuery table based on FHRSID.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        fhrsid: The FHRSID value to identify the rows to update.
        update_data: A dictionary where keys are column names and values are the new values.

    Returns:
        True if the update was successful, False otherwise.
    """
    if not update_data:
        print("No data provided for update.")
        return False

    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"

    set_clauses = []
    for column, value in update_data.items():
        if isinstance(value, str):
            # Escape single quotes within the string value itself
            sanitized_value = value.replace("'", "").replace('\\', '\\\\')
            set_clauses.append(f"`{column}` = '{sanitized_value}'")
        elif isinstance(value, bool):
            set_clauses.append(f"`{column}` = {str(value).upper()}")
        elif isinstance(value, (int, float)):
            set_clauses.append(f"`{column}` = {value}")
        elif value is None:
            set_clauses.append(f"`{column}` = NULL")
        else:
            # Fallback for other types, assuming string representation is acceptable
            # or specific handling might be needed for other types (e.g., DATE, TIMESTAMP)
            sanitized_value = str(value).replace("'", "").replace('\\', '\\\\')
            print(f"Warning: Column '{column}' has an unhandled type {type(value)}. Converting to string: '{sanitized_value}'")
            set_clauses.append(f"`{column}` = '{sanitized_value}'")


    if not set_clauses:
        print("No valid SET clauses generated from update_data.")
        return False

    set_statement = ", ".join(set_clauses)

    # FHRSID_COLNAME is used here. fhrsid value is already a string.
    # Ensure fhrsid value itself is escaped for single quotes if it can contain them,
    # though FHRSIDs are typically numeric strings and less likely to have quotes.
    escaped_fhrsid_value = fhrsid.replace("'", "''")
    query = f"UPDATE `{table_ref_str}` SET {set_statement} WHERE {FHRSID_COLNAME} = '{escaped_fhrsid_value}'"

    print(f"Executing BigQuery UPDATE query: {query}")

    try:
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        if query_job.errors:
            print(f"BigQuery UPDATE failed with errors: {query_job.errors}")
            # st.error(f"BigQuery UPDATE failed: {query_job.errors}") # Use st.error if in Streamlit context
            return False
        # Check if any rows were actually updated, if possible and relevant.
        # num_dml_affected_rows is not available for UPDATE in all cases or might need specific API versions.
        # For now, success is defined by no errors during execution.
        print(f"Successfully updated rows in {table_ref_str} for {FHRSID_COLNAME} = '{escaped_fhrsid_value}'.")
        # st.success(f"Successfully updated rows in {table_ref_str} for {FHRSID_COLNAME} = '{fhrsid}'.")
        return True
    except DefaultCredentialsError as e:
        print(f"BigQuery authentication error: {e}. Ensure your environment is configured correctly for ADC.")
        # st.error(f"BigQuery authentication error: {e}")
        return False
    except Exception as e:
        print(f"An error occurred during BigQuery UPDATE: {e}")
        # st.error(f"An error occurred during BigQuery UPDATE: {e}")
        return False
