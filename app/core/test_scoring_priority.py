import datetime
import math
import pandas as pd
import pytest
from app.core.data_processing import (
    haversine_distance_km,
    get_outcode_coordinates,
    lookup_outcode_coordinates,
    extract_outcode,
    calculate_restaurant_priority,
    LONDON_OUTCODE_CENTROIDS,
    UNKNOWN_OUTCODE,
    UNKNOWN_LOCATION_PROXIMITY_SCORE,
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

def test_a_missing_postcode_is_unknown_not_sw16():
    """D4. A blank postcode used to come back as the anchor outcode itself,
    which put the row 0 km from home and handed it the maximum proximity
    score -- missing data outranking every restaurant whose location we know."""
    assert extract_outcode("") == UNKNOWN_OUTCODE
    assert extract_outcode(None) == UNKNOWN_OUTCODE
    assert extract_outcode(float("nan")) == UNKNOWN_OUTCODE
    assert extract_outcode("   ") == UNKNOWN_OUTCODE

def test_lookup_returns_nothing_when_the_location_is_unresolvable():
    """The row path needs to tell "we know where this is" from "we do not".
    104 rows hold no postcode at all and two hold junk ('WATERLOOVI', 'NE');
    a centroid for those would be an invention."""
    assert lookup_outcode_coordinates("") is None
    assert lookup_outcode_coordinates("WATERLOOVI") is None
    assert lookup_outcode_coordinates("SW16 1AA") == tuple(LONDON_OUTCODE_CENTROIDS["SW16"])

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

def test_an_unknown_location_does_not_score_as_perfect_proximity():
    """The headline of D4: no postcode, no coordinates, top of the queue."""
    df = pd.DataFrame([
        {"fhrsid": "no-location", "postcode": None, "in_scope": True},
        {"fhrsid": "junk-postcode", "postcode": "WATERLOOVI", "in_scope": True},
    ])

    res = calculate_restaurant_priority(df)

    for _, row in res.iterrows():
        assert row["proximity_score"] == UNKNOWN_LOCATION_PROXIMITY_SCORE
        assert row["proximity_score"] < 100.0
        # Not 0.0 km either -- the UI shows a blank, not a fabricated distance.
        assert math.isnan(row["distance_km"])

def test_a_restaurant_we_can_place_outranks_one_we_cannot():
    """The queue-ordering consequence, stated as a comparison rather than a
    threshold: knowing where a restaurant is has to be worth something."""
    sw16_lat, sw16_lon = LONDON_OUTCODE_CENTROIDS["SW16"]
    df = pd.DataFrame([
        {"fhrsid": "nearby", "postcode": "SW16 1AA", "in_scope": True},
        {"fhrsid": "unknown", "postcode": "", "in_scope": True},
    ])

    res = calculate_restaurant_priority(df, anchor_lat=sw16_lat, anchor_lon=sw16_lon)
    scores = dict(zip(res["fhrsid"], res["priority_score"]))

    assert scores["nearby"] > scores["unknown"]

def test_an_unknown_location_still_beats_nothing():
    """...but not so far down that a row with no postcode can never be
    profiled. It keeps its staleness and scope weight; only proximity is
    discounted, to roughly what a restaurant 11 km away would score."""
    df = pd.DataFrame([{"fhrsid": "unknown", "postcode": "", "in_scope": True}])

    res = calculate_restaurant_priority(df)

    assert res.iloc[0]["priority_score"] > 0.0

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
        "gemini_profiled_at": "2026-06-20 12:00:00 UTC",
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
        "gemini_profiled_at": "2026-08-29 12:00:00 UTC",
        "maps_rating": 4.5,
        "maps_reviews": 100,
        "first_seen": "2026-01-01"
    }])

    res = calculate_restaurant_priority(df, today_date=today)
    row = res.iloc[0]
    # Stale score for <14 days should be 15.0
    assert row["staleness_score"] == 15.0

def test_staleness_reads_the_profile_timestamp_not_the_json_blob():
    """`gemini_insights` (V1 text) is NULL on every row, so the old
    `insights or structured` test always fell through to the raw blob. The
    stamp is the typed equivalent -- measured as exactly equivalent in Phase 7
    -- and it outlives the Phase 10 column drop."""
    today = datetime.date(2026, 8, 31)
    df = pd.DataFrame([{
        "fhrsid": "501",
        "postcode": "SW16 1AA",
        "in_scope": True,
        "predicted_user_rating": 7.0,
        "predicted_at": "2026-08-29 12:00:00 UTC",
        "gemini_profiled_at": None,
        "gemini_insights_structured": '{"match_score": 90}',
    }])

    res = calculate_restaurant_priority(df, today_date=today)

    # Unstamped reads as never profiled, blob or no blob.
    assert res.iloc[0]["staleness_score"] == 100.0

def test_a_profiled_and_predicted_row_is_not_scored_as_never_seen():
    """D-18, and the reason the switch above is not cosmetic. BigQuery NULLs
    arrive as `float('nan')` in a DataFrame, NaN is truthy, so
    `gemini_insights or gemini_insights_structured` stopped at the missing V1
    text and returned NaN. 1,020 live rows carried a profile and a prediction
    and were still being queued as if they had neither.

    The reproducing column is gone as of Phase 10, so this can no longer set
    the trap it was written for; the NaN-truthy lesson lives on in the
    postcode test below. What it still asserts is the behaviour the defect
    denied: a row with a profile and a recent prediction gets its real tier."""
    today = datetime.date(2026, 8, 31)
    df = pd.DataFrame([{
        "fhrsid": "601",
        "postcode": "SW16 1AA",
        "in_scope": True,
        "predicted_user_rating": 7.0,
        "predicted_at": "2026-08-10 12:00:00 UTC",   # 21 days -> the 40.0 tier
        "gemini_insights_structured": '{"match_score": 90}',
        "gemini_profiled_at": "2026-08-10 12:00:00 UTC",
    }])

    res = calculate_restaurant_priority(df, today_date=today)

    assert res.iloc[0]["staleness_score"] == 40.0

def test_a_nan_postcode_is_unknown_rather_than_the_string_nan():
    """The same NaN-is-truthy trap one component up: `row.get('postcode') or
    row.get('PostCode')` returned NaN, and `str(nan)` is `'nan'`, which the
    old centroid lookup happily resolved to central London."""
    df = pd.DataFrame([{"fhrsid": "701", "postcode": float("nan"),
                        "PostCode": None, "in_scope": True}])

    res = calculate_restaurant_priority(df)

    assert res.iloc[0]["proximity_score"] == UNKNOWN_LOCATION_PROXIMITY_SCORE

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
