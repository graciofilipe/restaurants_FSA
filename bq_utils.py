import streamlit as st
import pandas as pd
from google.cloud import bigquery
from typing import List
import re

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
        job.result()  # Wait for the job to complete
        st.success(f"Successfully wrote data to BigQuery table {table_ref_str} with schema and sanitized column names. Overwritten if table existed.")
        return True
    except Exception as e:
        st.error(f"Error writing data to BigQuery table {table_ref_str}: {e}")
        return False
