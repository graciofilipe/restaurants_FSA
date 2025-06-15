import streamlit as st
import pandas as pd
from google.cloud import bigquery
# from bq_utils import get_recent_restaurants # Assuming this is still needed elsewhere or will be handled
import bq_utils # Assuming this is still needed elsewhere or will be handled


def call_gemini_with_fhrs_data(project_id: str, dataset_id: str, gemini_prompt: str) -> pd.DataFrame:
    """
    Uses BigQuery AI.GENERATE to get insights for restaurants from 'recent_restaurants_temp'
    and stores them in 'genairesults_temp'. Then fetches these results.

    Args:
        project_id (str): Google Cloud project ID.
        dataset_id (str): BigQuery dataset ID.
        gemini_prompt (str): The base prompt for Gemini.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'fhrsid' and 'gemini_insights'.
                      Returns an empty DataFrame if no insights are found or an error occurs.
    """
    
    client = bigquery.Client(project=project_id)

    genairesults_temp_table_full_id = f"{project_id}.{dataset_id}.genairesults_temp"
    recent_restaurants_temp_table_full_id = f"{project_id}.{dataset_id}.recent_restaurants_temp"

    model_params_json_lit = "JSON '''{ \"tools\": [{\"googleSearch\": {}}], \"generationConfig\": { \"temperature\": 0.2, \"maxOutputTokens\": 8192, \"topP\": 1, \"seed\": 0 } }'''"

    # This SQL query assumes that the table referenced by `recent_restaurants_temp_table_full_id`
    # (i.e., `recent_restaurants_temp`) already has a column named `gemini_insights`.
    # If `gemini_insights` does not exist in `recent_restaurants_temp`, this query will fail.
    # The original issue's query included `WHERE gemini_insights IS NULL`.
    sql_query_create_results = f"""
    CREATE OR REPLACE TABLE `{genairesults_temp_table_full_id}` AS
    SELECT
      fhrsid,
      AI.GENERATE(
        ( '{gemini_prompt}',
          businessname,
          ' in ',
          addressline1,
          ' ',
          addressline2,
          ' ',
          addressline3,
          ' ',
          postcode,
          '. Use the results from Google Search and do not infer based on other knowledge.'
        ),
        connection_id => 'eu.gemini',
        endpoint => 'gemini-2.0-flash-001',
        model_params => {model_params_json_lit}
      ).result AS gemini_insights,
      businessname,
      addressline1,
      addressline2,
      addressline3,
      postcode
    FROM
      `{recent_restaurants_temp_table_full_id}`
    WHERE
      gemini_insights IS NULL OR gemini_insights = ''
    """

    # st.info(f"Executing BigQuery job to generate Gemini insights into: {genairesults_temp_table_full_id}") # Replaced by more specific message below
    st.info(f"Starting BigQuery job to generate Gemini insights. This may take a few moments...")
    try:
        print(f"Executing Gemini generation SQL: {sql_query_create_results}")
        query_job_create = client.query(sql_query_create_results)
        query_job_create.result()  # Wait for the job to complete

        if query_job_create.num_dml_affected_rows is not None:
            st.success(f"Gemini insights generation job complete. {query_job_create.num_dml_affected_rows} rows processed into {genairesults_temp_table_full_id}.")
        else:
            # Fallback message if num_dml_affected_rows is None (e.g., for CREATE OR REPLACE)
            # Attempt to get the number of rows in the created table as an alternative.
            try:
                table_info = client.get_table(genairesults_temp_table_full_id)
                st.success(f"Gemini insights generation job complete. '{genairesults_temp_table_full_id}' created/updated with {table_info.num_rows} rows.")
            except Exception:
                 st.success(f"Successfully created/updated '{genairesults_temp_table_full_id}' with Gemini insights.")


        # Now, fetch the results from the newly created table
        sql_query_select_results = f"SELECT fhrsid, gemini_insights FROM `{genairesults_temp_table_full_id}` WHERE gemini_insights IS NOT NULL AND gemini_insights != ''"
        st.info(f"Fetching generated insights from {genairesults_temp_table_full_id}...")
        print(f"Executing SQL to fetch results: {sql_query_select_results}")

        query_job_select = client.query(sql_query_select_results)
        results_df = query_job_select.to_dataframe()
        print(f"Fetched insights DataFrame shape: {results_df.shape}")

        if not results_df.empty:
            st.success(f"Successfully fetched {len(results_df)} insights from {genairesults_temp_table_full_id}.")
        else:
            st.warning(f"No insights were generated or found in {genairesults_temp_table_full_id}.")

        return results_df

    except Exception as e:
        st.error(f"An error occurred during BigQuery operations or data fetching: {e}")
        print(f"Error in call_gemini_with_fhrs_data: {e}") # For backend logging
        return pd.DataFrame() # Return empty DataFrame on error
 

# Helper function to map pandas dtypes to BigQuery types
def pandas_dtype_to_bq_type(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return 'INTEGER'
    elif pd.api.types.is_float_dtype(dtype):
        return 'FLOAT'
    elif pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    else:
        return 'STRING'


def create_recent_restaurants_temp_table(restaurants_df: pd.DataFrame, project_id: str, dataset_id: str):
    """
    Writes the provided DataFrame to a BigQuery table named "recent_restaurants_temp".

    Args:
        restaurants_df (pd.DataFrame): DataFrame containing restaurant data.
        project_id (str): Google Cloud project ID.
        dataset_id (str): BigQuery dataset ID.
    """
    if restaurants_df is None or restaurants_df.empty:
        st.warning("No restaurant data provided to create_recent_restaurants_temp_table. Skipping table creation.")
        return

    try:
        # Define columns_to_select (all columns from the DataFrame)
        columns_to_select = restaurants_df.columns.tolist()

        # Infer bq_schema from DataFrame dtypes
        bq_schema = []
        for column in restaurants_df.columns:
            bq_schema.append(
                bigquery.SchemaField(name=column, field_type=pandas_dtype_to_bq_type(restaurants_df[column].dtype))
            )

        st.write("Inferred BigQuery Schema for temporary table:", bq_schema) # For debugging or info

        # Call write_to_bigquery
        table_id_temp = "recent_restaurants_temp"

        # bq_utils is imported at the top of the file
        success = bq_utils.write_to_bigquery(
            df=restaurants_df,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id_temp,
            columns_to_select=columns_to_select,
            bq_schema=bq_schema
        )

        if success:
            st.success(f"Successfully wrote data to BigQuery temporary table: {project_id}.{dataset_id}.{table_id_temp}")
        else:
            st.error(f"Failed to write data to BigQuery temporary table: {project_id}.{dataset_id}.{table_id_temp}")

    except Exception as e:
        st.error(f"An error occurred during create_recent_restaurants_temp_table: {e}")