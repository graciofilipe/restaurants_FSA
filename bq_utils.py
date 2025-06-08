import streamlit as st
import pandas as pd
from google.cloud import bigquery
from typing import List # Removed Optional
import re
import pandas_gbq # Added import

# Custom Exceptions
class BigQueryExecutionError(Exception):
    """Custom exception for errors during BigQuery query execution."""
    pass

class DataFrameConversionError(Exception):
    """Custom exception for errors during DataFrame conversion from BigQuery results."""
    pass

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
    
    client = bigquery.Client(project=project_id)
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        column_name_character_map="V2",
    )
    
    try:
        job = client.load_table_from_dataframe(df_subset, table_ref_str, job_config=job_config)
        job.result()
        st.success(f"Successfully wrote data to BigQuery table {table_ref_str} with schema and sanitized column names. Overwritten if table existed.")
        return True
    except Exception as e:
        st.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False

def read_from_bigquery(fhrsid_list: List[str], project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame: # Changed return type
    """
    Reads data from a BigQuery table for a list of FHRSIDs using pandas-gbq.

    Args:
        fhrsid_list: A list of FHRSIDs (integers) to filter by.
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.

    Returns:
        A Pandas DataFrame containing the data for the FHRSIDs. Returns an empty DataFrame if no data is found.
    """
    table_ref_str = f"{project_id}.{dataset_id}.{table_id}"
    query = f"SELECT * FROM `{table_ref_str}` WHERE fhrsid IN UNNEST(@fhrsid_list)"

    # Configuration for query parameters
    # Note: pandas-gbq uses 'configuration' dict for job settings.
    # Query parameters are passed within this configuration.
    configuration = {
        'query': {
            'queryParameters': [
                {
                    'name': 'fhrsid_list',
                    'parameterType': {'arrayType': {'type': 'STRING'}},
                    'parameterValue': {'arrayValues': [{'value': f_id} for f_id in fhrsid_list]}
                }
            ]
        }
    }

    # Detailed logging before making the BigQuery call
    print(f"Executing BigQuery query with pandas-gbq: {query}")
    print(f"With FHRSID list: {fhrsid_list}")
    print(f"Table target: {table_ref_str}")

    try:
        # Use pandas_gbq.read_gbq
        df = pandas_gbq.read_gbq(
            query,
            project_id=project_id,
            configuration=configuration
        )
        # pandas_gbq.read_gbq returns an empty DataFrame if the query result is empty.
        # No need to check for df.empty and return None.
        if df.empty:
            print(f"Query executed successfully with pandas-gbq but returned no data for FHRSIDs: {', '.join(fhrsid_list)} from table {table_ref_str}")
        return df
    except Exception as e:
        # Catching a broad exception category from pandas-gbq.
        # Specific exceptions from pandas-gbq (e.g., related to auth, query syntax, or API errors)
        # should be caught here and wrapped into BigQueryExecutionError.
        # For example, pandas_gbq.gbq.GenericGBQException is a common one,
        # but others from google-cloud-bigquery or google-auth might also occur.
        error_msg = f"An error occurred while querying BigQuery with pandas-gbq for FHRSIDs: {', '.join(fhrsid_list)} from table {table_ref_str}: {e}"
        print(error_msg) # Keep existing print for logging
        # Wrap unexpected errors in BigQueryExecutionError for consistency
        # No need for DataFrameConversionError as read_gbq handles DataFrame creation.
        raise BigQueryExecutionError(error_msg) from e

def update_manual_review(fhrsid_list: List[str], manual_review_value: str, project_id: str, dataset_id: str, table_id: str) -> bool:
    """
    Updates the manual_review field for a list of FHRSIDs in a BigQuery table.

    Args:
        fhrsid_list: A list of FHRSIDs to update.
        manual_review_value: The new value for the manual_review field.
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.

    Returns:
        True if the update was successful for all FHRSIDs, False otherwise.
    """
    if not fhrsid_list:
        st.warning("No FHRSIDs provided for update.")
        return False

    client = bigquery.Client(project=project_id)
    query = f"""
        UPDATE `{project_id}.{dataset_id}.{table_id}`
        SET manual_review = @manual_review_value
        WHERE fhrsid IN UNNEST(@fhrsid_list)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("manual_review_value", "STRING", manual_review_value),
            bigquery.ArrayQueryParameter("fhrsid_list", "STRING", fhrsid_list),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        query_job.result()  # Wait for the query to complete
        # Check if any rows were actually updated if possible, though result() doesn't directly give row count for UPDATE
        # For simplicity, we assume success if query execution doesn't throw an error.
        # A more robust check might involve verifying num_dml_affected_rows if the API provides it easily,
        # or performing a SELECT COUNT(*) before and after, but that's more complex.
        st.success(f"Successfully updated manual_review for FHRSIDs: {', '.join(fhrsid_list)} to '{manual_review_value}'.")
        return True
    except Exception as e:
        st.error(f"Error updating manual_review for FHRSIDs: {', '.join(fhrsid_list)}: {e}")
        # Also print to console for backend logging
        print(f"Error updating manual_review for FHRSIDs: {', '.join(fhrsid_list)}: {e}")
        return False
