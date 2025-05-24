import requests
import streamlit as st
from typing import Optional, Dict, Any

def fetch_api_data(longitude: float, latitude: float, max_results: int) -> Optional[Dict[str, Any]]:
    """
    Fetches data from the Food Standards Agency API.

    Args:
        longitude: The longitude for the API search.
        latitude: The latitude for the API search.
        max_results: The maximum number of results to fetch from the API.

    Returns:
        A dictionary containing the JSON response from the API, or None if an error occurs.
    """
    api_url = f"https://api1-ratings.food.gov.uk/enhanced-search/en-GB/%5e/%5e/DISTANCE/1/Englad/{longitude}/{latitude}/1/{max_results}/json"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: Could not fetch data from the API. Status Code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: An exception occurred while making the API request: {e}")
        return None
