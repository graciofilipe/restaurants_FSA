import pandas as pd
import pytest
from app.ui.st_app import filter_and_sort_restaurants, display_data, get_selected_rows

@pytest.fixture
def sample_restaurants_df():
    return pd.DataFrame([
        {
            "fhrsid": "101",
            "businessname": "Pizza Palace",
            "postcode": "EC1A 1BB",
            "localauthorityname": "Islington",
            "in_scope": True,
            "user_rating": 8.0,
            "predicted_user_rating": 8.5,
            "maps_rating": 4.5,
            "match_score": 92.0,
            "first_seen": "2026-01-01",
        },
        {
            "fhrsid": "102",
            "businessname": "Burger Barn",
            "postcode": "W1D 4EQ",
            "localauthorityname": "Westminster",
            "in_scope": True,
            "user_rating": None,
            "predicted_user_rating": 7.2,
            "maps_rating": 4.1,
            "match_score": None,
            "first_seen": "2026-02-01",
        },
        {
            "fhrsid": "103",
            "businessname": "Coffee Corner",
            "postcode": "E1 6AN",
            "localauthorityname": "Tower Hamlets",
            "in_scope": False,
            "user_rating": None,
            "predicted_user_rating": None,
            "maps_rating": None,
            "match_score": None,
            "first_seen": "2026-03-01",
        },
        {
            "fhrsid": "104",
            "businessname": "Taco Town",
            "postcode": "N1 9AL",
            "localauthorityname": "Islington",
            "in_scope": None,
            "user_rating": None,
            "predicted_user_rating": 6.0,
            "maps_rating": None,
            "match_score": 75.0,
            "first_seen": "2026-04-01",
        },
    ])

def test_filter_empty_dataframe():
    empty_df = pd.DataFrame()
    res = filter_and_sort_restaurants(empty_df)
    assert res.empty

def test_filter_scope(sample_restaurants_df):
    in_scope = filter_and_sort_restaurants(sample_restaurants_df, scope_filter="In-Scope (Restaurants)")
    assert len(in_scope) == 2
    assert set(in_scope["fhrsid"]) == {"101", "102"}

    out_scope = filter_and_sort_restaurants(sample_restaurants_df, scope_filter="Out-of-Scope")
    assert len(out_scope) == 1
    assert out_scope.iloc[0]["fhrsid"] == "103"

    triage = filter_and_sort_restaurants(sample_restaurants_df, scope_filter="Unprocessed / Triage")
    assert len(triage) == 1
    assert triage.iloc[0]["fhrsid"] == "104"

def test_filter_user_rating_status(sample_restaurants_df):
    unrated = filter_and_sort_restaurants(sample_restaurants_df, user_rating_filter="No User Rating (Unrated)")
    assert len(unrated) == 3
    assert "101" not in unrated["fhrsid"].values

    rated = filter_and_sort_restaurants(sample_restaurants_df, user_rating_filter="Has User Rating (Rated)")
    assert len(rated) == 1
    assert rated.iloc[0]["fhrsid"] == "101"

def test_filter_ml_predictions(sample_restaurants_df):
    pred_only = filter_and_sort_restaurants(sample_restaurants_df, pred_rating_filter="Has Predicted Rating", min_pred_score=7.0)
    assert len(pred_only) == 2
    assert set(pred_only["fhrsid"]) == {"101", "102"}

    unpredicted = filter_and_sort_restaurants(sample_restaurants_df, pred_rating_filter="No Predicted Rating")
    assert len(unpredicted) == 1
    assert unpredicted.iloc[0]["fhrsid"] == "103"

def test_filter_gemini_match_score(sample_restaurants_df):
    has_gemini = filter_and_sort_restaurants(sample_restaurants_df, gemini_match_filter="Has Gemini Match Score")
    assert len(has_gemini) == 2
    assert set(has_gemini["fhrsid"]) == {"101", "104"}

    no_gemini = filter_and_sort_restaurants(sample_restaurants_df, gemini_match_filter="No Gemini Match Score")
    assert len(no_gemini) == 2
    assert set(no_gemini["fhrsid"]) == {"102", "103"}

def test_filter_maps_rating(sample_restaurants_df):
    has_maps = filter_and_sort_restaurants(sample_restaurants_df, maps_rating_filter="Has Google Maps Rating")
    assert len(has_maps) == 2
    assert set(has_maps["fhrsid"]) == {"101", "102"}

    no_maps = filter_and_sort_restaurants(sample_restaurants_df, maps_rating_filter="No Google Maps Rating")
    assert len(no_maps) == 2
    assert set(no_maps["fhrsid"]) == {"103", "104"}

def test_search_query_name_and_postcode(sample_restaurants_df):
    by_name = filter_and_sort_restaurants(sample_restaurants_df, search_query="pizza")
    assert len(by_name) == 1
    assert by_name.iloc[0]["businessname"] == "Pizza Palace"

    by_postcode = filter_and_sort_restaurants(sample_restaurants_df, search_query="W1D")
    assert len(by_postcode) == 1
    assert by_postcode.iloc[0]["fhrsid"] == "102"

def test_sorting_options(sample_restaurants_df):
    # Sort by Predicted Rating High to Low
    sorted_pred = filter_and_sort_restaurants(sample_restaurants_df, sort_by="Predicted Rating (High to Low)")
    assert sorted_pred.iloc[0]["fhrsid"] == "101" # 8.5
    assert sorted_pred.iloc[1]["fhrsid"] == "102" # 7.2
    assert sorted_pred.iloc[2]["fhrsid"] == "104" # 6.0
    assert sorted_pred.iloc[3]["fhrsid"] == "103" # None

    # Sort by Maps Rating High to Low
    sorted_maps = filter_and_sort_restaurants(sample_restaurants_df, sort_by="Maps Rating (High to Low)")
    assert sorted_maps.iloc[0]["fhrsid"] == "101" # 4.5
    assert sorted_maps.iloc[1]["fhrsid"] == "102" # 4.1

    # Sort by Gemini Match Score High to Low
    sorted_gemini = filter_and_sort_restaurants(sample_restaurants_df, sort_by="Gemini Match Score (High to Low)")
    assert sorted_gemini.iloc[0]["fhrsid"] == "101" # 92.0
    assert sorted_gemini.iloc[1]["fhrsid"] == "104" # 75.0

    # Sort by Business Name A-Z
    sorted_name = filter_and_sort_restaurants(sample_restaurants_df, sort_by="Business Name (A-Z)")
    assert list(sorted_name["businessname"]) == ["Burger Barn", "Coffee Corner", "Pizza Palace", "Taco Town"]

    # Sort by First Seen Newest
    sorted_date = filter_and_sort_restaurants(sample_restaurants_df, sort_by="First Seen (Newest)")
    assert sorted_date.iloc[0]["fhrsid"] == "104" # 2026-04-01

