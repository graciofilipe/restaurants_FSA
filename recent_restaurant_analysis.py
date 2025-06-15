import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
from bq_utils import get_recent_restaurants
from google.cloud import bigquery # Added import

N_DAYS = 1


def call_gemini_with_fhrs_data(fhrs_ids, gemini_prompt, df):
    """
    For each FHRSID, this function calls Vertex AI Gemini with a prompt that includes
    the restaurant's name and address, using Google Search for grounding.
    This version uses the genai.Client() initialization pattern.

    Args:
        fhrs_ids (list): A list of FHRSIDs.
        gemini_prompt (str): The base Gemini prompt.
        df (pd.DataFrame): The DataFrame containing restaurant data.

    Returns:
        None: Prints the Gemini response for each FHRSID.
    """
    # 1. Initialize the client to use Vertex AI, specifying project and location.
    # This is the new, recommended pattern.
    client = genai.Client(
        vertexai=True,
        project="filipegracio-ai-learning",
        location="us-central1",  # A specific region is required, 'global' is not supported for this.
    )

    # 2. Define the model and the tool configuration.
    # Using a recent preview model as per your example.
    model_name = "gemini-2.5-flash-preview-05-20"
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]

    # Define the generation configuration, including the tools.
    generation_config = types.GenerateContentConfig(tools=tools)
    results_list=[]
    for fhrs_id in fhrs_ids:
        restaurant_data = df[df['fhrsid'] == fhrs_id]

        if not restaurant_data.empty:
            restaurant_name = restaurant_data['businessname'].iloc[0]
            address_fields = ["addressline1", "addressline2", "addressline3", "postcode", "localauthorityname"]
            address_info = ", ".join(
                str(restaurant_data[field].iloc[0])
                for field in address_fields
                if pd.notna(restaurant_data[field].iloc[0]) and str(restaurant_data[field].iloc[0]).strip()
            )

            modified_prompt = f"{gemini_prompt} \n Restaurant Name: {restaurant_name}, \n Address: {address_info}"

            # 3. Structure the prompt using the types.Content and types.Part format.
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=modified_prompt)]
                ),
            ]
            n = len(results_list)
            st.info(f"--- Querying for fhrsid: {fhrs_id} ({restaurant_name}) , number {n}---")

            # 4. Call the model using the client, passing all configurations.
            # Note: The SDK doesn't support a direct 'chat' mode with the Client interface.
            # We send the full content history with each request.
            response = client.models.generate_content(
                model=model_name,  # The model name needs the 'models/' prefix here
                contents=contents,
                config=generation_config,
            )

            # --- CHANGED: Instead of printing, append the result to our list ---
            results_list.append({
                  'fhrsid': fhrs_id,
                  'gemini_insights': response.text.strip()
            })

        else:
            st.warning(f"No data found for fhrsid: {fhrs_id}")

        results_df = pd.DataFrame(results_list)

    return results_df


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