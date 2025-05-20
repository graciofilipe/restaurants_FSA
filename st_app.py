import streamlit as st
import requests
import json
import pandas as pd
from google.cloud import storage
from datetime import datetime
import os

# Set the title of the Streamlit app
st.title("Food Standards Agency API Explorer")

# Create input fields for longitude and latitude
longitude = st.number_input("Enter Longitude", format="%.6f")
latitude = st.number_input("Enter Latitude", format="%.6f")

# Create an input field for the GCS destination folder URI
gcs_destination_uri = st.text_input("Enter GCS Destination Folder URI (e.g., gs://bucket-name/folder-name)")

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

            # GCS Upload Logic
            if gcs_destination_uri: # Check if GCS URI is provided
                if not gcs_destination_uri.startswith("gs://"):
                    st.error("Invalid GCS URI. It must start with gs://")
                else:
                    try:
                        # Construct the filename
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        file_name = f"food_standards_data_{current_date}.json"

                        # Initialize GCS client
                        storage_client = storage.Client()

                        # Parse GCS URI
                        bucket_name = gcs_destination_uri.split("/")[2]
                        blob_name_prefix = "/".join(gcs_destination_uri.split("/")[3:])
                        
                        # Construct the full blob path
                        blob_path = os.path.join(blob_name_prefix, file_name)

                        # Get bucket and create blob
                        bucket = storage_client.bucket(bucket_name)
                        blob = bucket.blob(blob_path)

                        # Upload data
                        blob.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
                        st.success(f"Successfully uploaded to gs://{bucket_name}/{blob_path}")
                    except Exception as e:
                        st.error(f"Error uploading to GCS: {e}")
            
            try:
                establishments = data['FHRSEstablishment']['EstablishmentCollection']['EstablishmentDetail']
                df = pd.json_normalize(establishments)
                st.dataframe(df)
            except KeyError:
                st.error("Error: Could not find the expected data structure in the API response.")
            except TypeError: # Handles cases where establishments might be None if the key path is valid but no data
                st.warning("No establishment data found in the response, or the data format is unexpected.")

            # Display a download button for the JSON data
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
