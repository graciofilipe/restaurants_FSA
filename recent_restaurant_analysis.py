from google import genai
from google.genai import types
import pandas as pd
from bq_utils import get_recent_restaurants

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
            print(f"--- Querying for fhrsid: {fhrs_id} ({restaurant_name}) , number {n}---")

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
            print(f"No data found for fhrsid: {fhrs_id}")

        results_df = pd.DataFrame(results_list)

    return results_df


new_df = get_recent_restaurants(N_DAYS=N_DAYS, project_id=, dataset_id=, table_id=)

# only run the gemini analysis on the restaurants where the gemini_insights column is empty:
new_df = new_df[new_df['gemini_insights'].isnull()]

fhrs_ids_list = new_df['fhrsid'].tolist()
gemini_prompt = "Be succint and tell me what cuisine and dishes this specific London restaurant serve. \
    Do not infer from the name of the restaurant, and base your answer on what you find in your search. \n \
    Here is the Restaurant information: "

gemini_results_df = call_gemini_with_fhrs_data(fhrs_ids_list, gemini_prompt, new_df)

# Merge the two dataframes on 'fhrsid'
new_and_gemini_merged_df = pd.merge(new_df, gemini_results_df, on='fhrsid', how='left')

# assume most will be rejected and attribute this value by default
new_and_gemini_merged_df["manual_review"] = "rejected"

# display the new_and_gemini_merged_df dataframes on the streamlit app