import datetime
import pandas as pd
import pytest
from app.core.data_processing import (
    haversine_distance_km,
    get_outcode_coordinates,
    calculate_restaurant_priority,
    LONDON_OUTCODE_CENTROIDS,
)

def test_haversine_distance_km_zero_and_known():
    # Distance from SW16 centroid to itself
    lat, lon = LONDON_OUTCODE_CENTROIDS["SW16"]
    assert haversine_distance_km(lat, lon, lat, lon) == 0.0

    # Distance between SW16 (51.4277, -0.1294) and EC1 (51.5230, -0.0980) is approx 10.8 km
    lat_ec1, lon_ec1 = LONDON_OUTCODE_CENTROIDS["EC1"]
    dist = haversine_distance_km(lat, lon, lat_ec1, lon_ec1)
    assert 10.0 <= dist <= 12.0

def test_get_outcode_coordinates():
    sw16_coords = get_outcode_coordinates("SW16")
    assert sw16_coords == (51.4277, -0.1294)

    sw16_full = get_outcode_coordinates("SW16 1AA")
    assert sw16_full == (51.4277, -0.1294)

    ec1_coords = get_outcode_coordinates("EC1")
    assert ec1_coords == (51.5230, -0.0980)

    # Empty fallback
    assert get_outcode_coordinates("") == (51.4277, -0.1294)

def test_calculate_restaurant_priority_unscored_nearby():
    today = datetime.date(2026, 8, 31)
    # Restaurant right in SW16, unscored, high maps rating, in_scope
    df = pd.DataFrame([{
        "fhrsid": "101",
        "businessname": "Local Star Bistro",
        "postcode": "SW16 1AA",
        "latitude": 51.4277,
        "longitude": -0.1294,
        "in_scope": True,
        "predicted_user_rating": None,
        "gemini_insights": None,
        "gemini_insights_structured": None,
        "maps_rating": 4.8,
        "maps_reviews": 150,
        "first_seen": "2026-08-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    assert len(res) == 1
    row = res.iloc[0]
    
    # Distance should be 0 km
    assert row["distance_km"] == 0.0
    # Proximity score should be 100
    assert row["proximity_score"] == 100.0
    # Staleness score for unscored should be 100
    assert row["staleness_score"] == 100.0
    # Scope score should be 100
    assert row["priority_score"] >= 90.0

def test_calculate_restaurant_priority_stale_rescore():
    today = datetime.date(2026, 8, 31)
    # Restaurant scored 70 days ago (first_seen 2026-06-20)
    df = pd.DataFrame([{
        "fhrsid": "202",
        "businessname": "Historic Diner",
        "postcode": "SW16 2BB",
        "latitude": 51.4277,
        "longitude": -0.1294,
        "in_scope": True,
        "predicted_user_rating": 6.5,
        "gemini_insights": "Old insight",
        "gemini_insights_structured": '{"match_score": 75}',
        "maps_rating": 4.2,
        "maps_reviews": 80,
        "first_seen": "2026-06-20"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    # Stale score for >=60 days should be 80.0
    assert row["staleness_score"] == 80.0
    assert row["priority_score"] > 70.0

def test_calculate_restaurant_priority_out_of_scope_penalty():
    today = datetime.date(2026, 8, 31)
    df = pd.DataFrame([{
        "fhrsid": "303",
        "businessname": "Corner Bakery",
        "postcode": "SW16 3CC",
        "latitude": 51.4277,
        "longitude": -0.1294,
        "in_scope": False,
        "predicted_user_rating": None,
        "gemini_insights": None,
        "gemini_insights_structured": None,
        "maps_rating": 4.0,
        "maps_reviews": 10,
        "first_seen": "2026-08-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    # Out of scope gets 0 scope score
    assert row["priority_score"] < 80.0

def test_calculate_restaurant_priority_empty_df():
    res = calculate_restaurant_priority(pd.DataFrame())
    assert res.empty
