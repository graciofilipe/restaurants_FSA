import streamlit as st
import requests
import json
import pandas as pd

# Set the title of the Streamlit app
st.title("Food Standards Agency API Explorer")

# Create input fields for longitude and latitude
longitude = st.number_input("Enter Longitude", format="%.6f")
latitude = st.number_input("Enter Latitude", format="%.6f")

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
