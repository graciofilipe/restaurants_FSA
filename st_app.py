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
gcs_destination_uri = st.text_input("Enter GCS Destination Folder URI (e.g., gs://bucket-name/folder-name)")

# Create an input field for the master restaurant list URI
master_list_uri = st.text_input("Enter Master Restaurant List URI (e.g., gs://bucket/file.json or /path/to/file.json)")

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

            # 1.a. Load Master List
            restaurants_master_list = []
            if master_list_uri:
                loaded_master_data = load_json_from_uri(master_list_uri)
                if loaded_master_data is not None:
                    if isinstance(loaded_master_data, dict) and \
                       'FHRSEstablishment' in loaded_master_data and \
                       'EstablishmentCollection' in loaded_master_data.get('FHRSEstablishment', {}) and \
                       'EstablishmentDetail' in loaded_master_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}):
                        restaurants_master_list = loaded_master_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                        if not restaurants_master_list: # handles None or empty list from .get
                             st.warning("Master list loaded but no establishments found within the expected structure (FHRSEstablishment.EstablishmentCollection.EstablishmentDetail).")
                        else:
                            st.info(f"Successfully loaded {len(restaurants_master_list)} establishments from master list (structured format).")
                    elif isinstance(loaded_master_data, list):
                        restaurants_master_list = loaded_master_data
                        if not restaurants_master_list:
                            st.warning("Master list loaded but it's an empty list.")
                        else:
                            st.info(f"Successfully loaded {len(restaurants_master_list)} establishments from master list (assumed list format).")
                    else:
                        st.warning(f"Master list loaded from {master_list_uri} but is not a recognized format (expected a dictionary with FHRSEstablishment structure or a list of establishments). Starting with an empty master list.")
                        restaurants_master_list = [] 
                else:
                    # load_json_from_uri already shows an error via st.error
                    st.warning("Failed to load master list or it was empty. Proceeding with an empty master list.")
                    restaurants_master_list = []
            else:
                st.info("No master list URI provided. Starting with an empty master list.")
            
            if not isinstance(restaurants_master_list, list):
                st.warning(f"restaurants_master_list was not a list after loading attempts (type: {type(restaurants_master_list)}). Resetting to an empty list.")
                restaurants_master_list = []

            # 1.b. Process API Response and Update Master List
            api_establishments = data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
            if api_establishments is None: 
                api_establishments = []
                st.warning("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
            elif not api_establishments:
                 st.info("API response contained no establishments in 'EstablishmentDetail'.")


            existing_fhrsid_set = {est['FHRSID'] for est in restaurants_master_list if isinstance(est, dict) and 'FHRSID' in est}
            today_date = datetime.now().strftime("%Y-%m-%d")
            new_restaurants_added_count = 0

            for api_establishment in api_establishments:
                if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment:
                    if api_establishment['FHRSID'] not in existing_fhrsid_set:
                        api_establishment['first_seen'] = today_date
                        restaurants_master_list.append(api_establishment)
                        existing_fhrsid_set.add(api_establishment['FHRSID']) # Add to set to prevent duplicates from API response itself
                        new_restaurants_added_count += 1
            
            st.success(f"Processed API response. Added {new_restaurants_added_count} new restaurants. Total unique establishments: {len(restaurants_master_list)}")

            # GCS Upload Logic - uses original 'data' from API
            if gcs_destination_uri: 
                if not gcs_destination_uri.startswith("gs://"):
                    st.error("Invalid GCS URI. It must start with gs://")
                else:
                    try:
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        file_name = f"food_standards_data_{current_date}.json" # Original API data
                        storage_client = storage.Client()
                        bucket_name = gcs_destination_uri.split("/")[2]
                        blob_name_prefix = "/".join(gcs_destination_uri.split("/")[3:])
                        blob_path = os.path.join(blob_name_prefix, file_name)
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_path)
                        blob.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
                        st.success(f"Successfully uploaded raw API data to gs://{bucket_name}/{blob_path}")
                    except Exception as e:
                        st.error(f"Error uploading raw API data to GCS: {e}")
            
            # 2. Modify DataFrame Creation - uses 'restaurants_master_list'
            try:
                if not restaurants_master_list:
                    st.warning("No establishment data to display (master list is empty after processing).")
                else:
                    # Ensure all items are dictionaries before normalization
                    valid_items_for_df = [item for item in restaurants_master_list if isinstance(item, dict)]
                    if not valid_items_for_df:
                        st.warning("Master list contains no dictionary items, cannot display as table.")
                    elif len(valid_items_for_df) < len(restaurants_master_list):
                        st.warning(f"Some items in the master list were not dictionaries and were excluded from the table display. Displaying {len(valid_items_for_df)} items.")
                        df = pd.json_normalize(valid_items_for_df)
                        st.dataframe(df)
                    else:
                        df = pd.json_normalize(restaurants_master_list)
                        st.dataframe(df)
            except Exception as e: 
                st.error(f"Error displaying DataFrame from master list: {e}")
                st.info("Attempting to show raw master list as JSON.")
                try:
                    st.json(restaurants_master_list)
                except Exception as json_e:
                    st.error(f"Could not even display master list as JSON: {json_e}")

            # Display a download button for the JSON data - uses original 'data' from API
            st.download_button(
                label="Download JSON Data",
                data=json.dumps(data, indent=4),
                file_name="food_standards_data.json",
                mime="application/json",
            )
        else:
            # Display an error message if the API request fails
            st.error(f"Error: Could not fetch data from the API. Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error: An exception occurred while making the API request: {e}")
