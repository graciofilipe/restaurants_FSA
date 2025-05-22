import streamlit as st
import requests
import json
import pandas as pd
from google.cloud import storage
from datetime import datetime
import os


def load_json_from_uri(uri: str):
    """
    Loads a JSON file from a given URI (GCS path or local file path).

    Args:
        uri: The URI of the JSON file (e.g., "gs://bucket/file.json" or "/path/to/file.json").

    Returns:
        A dictionary loaded from the JSON file, or None if an error occurs.
    """
    if uri.startswith("gs://"):
        try:
            storage_client = storage.Client()
            bucket_name = uri.split("/")[2]
            blob_name = "/".join(uri.split("/")[3:])
            
            if not blob_name: # Handle case where URI might be just gs://bucket-name
                st.error(f"Invalid GCS URI: No file specified in gs://{bucket_name}/")
                return None

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                st.error(f"Error: GCS file not found at {uri}")
                return None

            data_string = blob.download_as_string()
            return json.loads(data_string)
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from GCS file {uri}: {e}")
            return None
        except Exception as e: # Catching a broader range of GCS client/permission errors
            st.error(f"Error accessing GCS file {uri}: {e}")
            return None
    else:
        try:
            with open(uri, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Error: Local file not found at {uri}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON from local file {uri}: {e}")
            return None
        except Exception as e:
            st.error(f"Error reading local file {uri}: {e}")
            return None

# Set the title of the Streamlit app
st.title("Food Standards Agency API Explorer")

# Create input fields for longitude and latitude
longitude = st.number_input("Enter Longitude", format="%.6f")
latitude = st.number_input("Enter Latitude", format="%.6f")

# Create an input field for the GCS destination folder URI
gcs_destination_uri = st.text_input("Enter GCS destination folder for the scan (e.g., gs://bucket-name/scans-folder/)")

# Create an input field for the master restaurant list URI
master_list_uri = st.text_input("Enter Master Restaurant Data URI (JSON) (e.g., gs://bucket/file.json or /path/to/file.json)")

# Create an input field for the GCS destination for the master restaurant data
gcs_master_dictionary_output_uri = st.text_input("Enter GCS URI for Master Restaurant Data Output (e.g., gs://bucket-name/path/filename.json)")

# Create a button to trigger the API call
if st.button("Fetch Data"):
    # Construct the API URL
    api_url = f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/500/json"

    try:
        # Make the GET request
        response = requests.get(api_url)

        # Check if the API request was successful
        if response.status_code == 200:
            # Store the JSON response
            data = response.json()

            # 1.a. Load Master Restaurant Data
            master_restaurant_data = []
            if master_list_uri:
                loaded_data = load_json_from_uri(master_list_uri)
                if loaded_data is not None:
                    if isinstance(loaded_data, list):
                        master_restaurant_data = loaded_data
                        if master_restaurant_data:
                            st.success(f"Successfully loaded master restaurant data with {len(master_restaurant_data)} records from {master_list_uri}.")
                        else:
                            st.warning(f"Master restaurant data loaded from {master_list_uri}, but it's empty.")
                    else:
                        # This case should ideally not happen if load_json_from_uri is robust
                        # and the JSON structure is expected to be a list at the root.
                        st.warning(f"Data loaded from {master_list_uri} is not in the expected format (e.g., a list of records). Type found: {type(loaded_data)}. Proceeding with empty master restaurant data.")
                        master_restaurant_data = []
                else:
                    # load_json_from_uri handles st.error for specific loading issues.
                    st.warning(f"Failed to load master restaurant data from {master_list_uri} (or it was empty/invalid). Proceeding with empty master restaurant data.")
                    master_restaurant_data = [] # Ensure it's an empty list on failure
            else:
                st.info("No master restaurant data URI provided. Starting with empty master restaurant data.")
                master_restaurant_data = [] # Ensure it's an empty list if no URI

            # 1.b. Process API Response and Update Master Restaurant Data
            api_establishments = data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
            if api_establishments is None: 
                api_establishments = []
                st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
            elif not api_establishments:
                 st.info("API response contained no establishments in 'EstablishmentDetail'.")


            existing_fhrsid_set = {est['FHRSID'] for est in master_restaurant_data if isinstance(est, dict) and 'FHRSID' in est}
            today_date = datetime.now().strftime("%Y-%m-%d")
            new_restaurants_added_count = 0

            for api_establishment in api_establishments:
                if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
                    if api_establishment['FHRSID'] not in existing_fhrsid_set:
                        api_establishment['first_seen'] = today_date
                        master_restaurant_data.append(api_establishment)
                        existing_fhrsid_set.add(api_establishment['FHRSID']) # Add to set to prevent duplicates from API response itself
                        new_restaurants_added_count += 1
            
            st.success(f"Processed API response. Added {new_restaurants_added_count} new restaurant records. Total unique records: {len(master_restaurant_data)}")

            # 1. Upload Raw API Response to gcs_destination_uri (folder)
            if gcs_destination_uri:
                if not gcs_destination_uri.startswith("gs://"):
                    st.error("Invalid GCS URI for API Response. It must start with gs://")
                else:
                    try:
                        storage_client = storage.Client() # Initialize client
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        api_response_filename = f"api_response_{current_date}.json"
                        
                        bucket_name_api = gcs_destination_uri.split("/")[2]
                        folder_path_api = "/".join(gcs_destination_uri.split("/")[3:])
                        
                        # Ensure folder_path_api ends with a slash if it's not empty
                        if folder_path_api and not folder_path_api.endswith('/'):
                            folder_path_api += '/'
                        
                        # Construct blob_path_api, removing leading slash if folder_path_api was empty
                        blob_path_api = f"{folder_path_api}{api_response_filename}"
                        if blob_path_api.startswith('/'): 
                           blob_path_api = blob_path_api[1:]

                        bucket_api = storage_client.bucket(bucket_name_api)
                        blob_api = bucket_api.blob(blob_path_api)
                        
                        blob_api.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
                        st.success(f"Successfully uploaded raw API response to gs://{bucket_name_api}/{blob_path_api}")
                    except Exception as e:
                        st.error(f"Error uploading raw API response to GCS: {e}")
            
            # 2. Upload Master Restaurant Data to gcs_master_dictionary_output_uri (full file path)
            if gcs_master_dictionary_output_uri:
                if not gcs_master_dictionary_output_uri.startswith("gs://"):
                    st.error("Invalid GCS URI for Master Restaurant Data. It must start with gs://")
                else:
                    try:
                        storage_client = storage.Client() # Initialize client
                        
                        bucket_name_master = gcs_master_dictionary_output_uri.split("/")[2]
                        blob_name_master = "/".join(gcs_master_dictionary_output_uri.split("/")[3:])

                        if not blob_name_master: 
                            st.error("Invalid GCS URI for Master Restaurant Data: File name is missing.")
                        else:
                            bucket_master = storage_client.bucket(bucket_name_master)
                            blob_master = bucket_master.blob(blob_name_master)
                            
                            blob_master.upload_from_string(json.dumps(master_restaurant_data, indent=4), content_type='application/json')
                            st.success(f"Successfully uploaded master restaurant data to {gcs_master_dictionary_output_uri}")
                    except Exception as e:
                        st.error(f"Error uploading master restaurant data to GCS: {e}")

            # DataFrame display logic
            try:
                if not master_restaurant_data:
                    st.warning("No restaurant data to display (master restaurant data is empty after processing).")
                else:
                    # Ensure all items are dictionaries before normalization
                    valid_items_for_df = [item for item in master_restaurant_data if isinstance(item, dict)]
                    if not valid_items_for_df:
                        st.warning("Master restaurant data contains no dictionary items, cannot display as table.")
                    elif len(valid_items_for_df) < len(master_restaurant_data):
                        st.warning(f"Some items in the master restaurant data were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} records.")
                        df = pd.json_normalize(valid_items_for_df)
                        st.dataframe(df)
                    else:
                        df = pd.json_normalize(master_restaurant_data)
                        st.dataframe(df)
            except Exception as e: 
                st.error(f"Error displaying DataFrame from master restaurant data: {e}")
                st.info("Attempting to show raw master restaurant data as JSON.")
                try:
                    st.json(master_restaurant_data)
                except Exception as json_e:
                    st.error(f"Could not even display master restaurant data as JSON: {json_e}")

        else:
            # Display an error message if the API request fails
            st.error(f"Error: Could not fetch data from the API. Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: An exception occurred while making the API request: {e}")
