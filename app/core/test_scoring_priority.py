import datetime
import pandas as pd
import pytest
from app.core.data_processing import (
    haversine_distance_km,
    get_outcode_coordinates,
    extract_outcode,
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

def test_extract_outcode():
    assert extract_outcode("SW4 7UL") == "SW4"
    assert extract_outcode("SW19 6NW") == "SW19"
    assert extract_outcode("EC2A 3DU") == "EC2A"
    assert extract_outcode("EC2A3DU") == "EC2A"
    assert extract_outcode("E27DJ") == "E2"
    assert extract_outcode("WD17 1AA") == "WD17"
    assert extract_outcode("") == "SW16"

def test_get_outcode_coordinates():
    sw16_c = LONDON_OUTCODE_CENTROIDS["SW16"]
    sw16_coords = get_outcode_coordinates("SW16")
    assert sw16_coords == sw16_c

    sw16_full = get_outcode_coordinates("SW16 1AA")
    assert sw16_full == sw16_c

    ec1_coords = get_outcode_coordinates("EC1")
    assert ec1_coords == LONDON_OUTCODE_CENTROIDS["EC1"]

    # Distant outcodes should not resolve to SW16
    wd17_coords = get_outcode_coordinates("WD17 1AA")
    assert wd17_coords != sw16_c

    cr7_coords = get_outcode_coordinates("CR7 8AA")
    assert cr7_coords != sw16_c

def test_outcode_distances_from_sw16():
    sw16_lat, sw16_lon = LONDON_OUTCODE_CENTROIDS["SW16"]
    df = pd.DataFrame([
        {"fhrsid": "1", "postcode": "SW16 1AA", "in_scope": True}, # 0 km
        {"fhrsid": "2", "postcode": "SE24 0JT", "in_scope": True}, # ~3.7 km
        {"fhrsid": "3", "postcode": "SW19 6NW", "in_scope": True}, # ~5.5 km
        {"fhrsid": "4", "postcode": "EC2A 3DU", "in_scope": True}, # ~10 km
        {"fhrsid": "5", "postcode": "WD17 1AA", "in_scope": True}, # ~30 km
    ])

    res = calculate_restaurant_priority(df, anchor_lat=sw16_lat, anchor_lon=sw16_lon)
    dists = dict(zip(res["fhrsid"], res["distance_km"]))
    
    assert dists["1"] == 0.0
    assert 2.0 <= dists["2"] <= 6.0
    assert 4.0 <= dists["3"] <= 8.0
    assert 9.0 <= dists["4"] <= 14.0
    assert dists["5"] >= 20.0

def test_calculate_restaurant_priority_unscored_nearby():
    today = datetime.date(2026, 8, 31)
    sw16_lat, sw16_lon = LONDON_OUTCODE_CENTROIDS["SW16"]
    # Restaurant right at SW16 centroid, unscored, high maps rating, in_scope
    df = pd.DataFrame([{
        "fhrsid": "101",
        "businessname": "Local Star Bistro",
        "postcode": "SW16 1AA",
        "latitude": sw16_lat,
        "longitude": sw16_lon,
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
        "predicted_at": "2026-06-20 12:00:00 UTC",
        "gemini_insights": "Old insight",
        "gemini_insights_structured": '{"match_score": 75}',
        "maps_rating": 4.2,
        "maps_reviews": 80,
        "first_seen": "2026-01-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    # Stale score for >=60 days should be 80.0
    assert row["staleness_score"] == 80.0
    assert row["priority_score"] > 70.0

def test_calculate_restaurant_priority_recent_prediction():
    today = datetime.date(2026, 8, 31)
    # Restaurant scored 2 days ago (predicted_at 2026-08-29)
    df = pd.DataFrame([{
        "fhrsid": "205",
        "businessname": "Fresh Diner",
        "postcode": "SW16 2BB",
        "latitude": 51.4277,
        "longitude": -0.1294,
        "in_scope": True,
        "predicted_user_rating": 8.5,
        "predicted_at": "2026-08-29 12:00:00 UTC",
        "gemini_insights": "Fresh insight",
        "gemini_insights_structured": '{"match_score": 90}',
        "maps_rating": 4.5,
        "maps_reviews": 100,
        "first_seen": "2026-01-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    # Stale score for <14 days should be 15.0
    assert row["staleness_score"] == 15.0

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

def test_calculate_restaurant_priority_user_rating_penalty():
    today = datetime.date(2026, 8, 31)
    df = pd.DataFrame([{
        "fhrsid": "401",
        "businessname": "Visited Steakhouse",
        "postcode": "SW16 1AA",
        "latitude": 51.4277,
        "longitude": -0.1294,
        "in_scope": True,
        "user_rating": 9.0,
        "predicted_user_rating": None,
        "gemini_insights": None,
        "gemini_insights_structured": None,
        "maps_rating": 4.8,
        "maps_reviews": 150,
        "first_seen": "2026-08-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    assert row["priority_score"] <= 10.0

def test_calculate_restaurant_priority_empty_df():
    res = calculate_restaurant_priority(pd.DataFrame())
    assert res.empty
