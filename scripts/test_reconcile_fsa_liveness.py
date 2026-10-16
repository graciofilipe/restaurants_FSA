"""Unit tests for `scripts/reconcile_fsa_liveness.py`."""

from unittest.mock import MagicMock

import requests

from scripts.reconcile_fsa_liveness import (
    build_candidates_query,
    build_carry_over_sql,
    build_link_superseded_sql,
    build_liveness_merge_sql,
    probe_establishment,
    reconcile_in_memory,
)


def _mock_response(status_code: int, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = b"{}" if json_body is not None else b""
    resp.json.return_value = json_body or {}
    return resp


def test_probe_establishment_200_and_404_and_5xx_retry():
    session = MagicMock()

    # 1. 200 live response
    session.get.return_value = _mock_response(
        200,
        {
            "FHRSID": 1871661,
            "BusinessName": "Taste Of Kerala",
            "AddressLine1": "305 London Road",
            "PostCode": "CR0 3PA",
            "RatingValue": "5",
            "NewRatingPending": False,
            "geocode": {"latitude": "51.3847167", "longitude": "-0.1094876"},
        },
    )
    res_200 = probe_establishment("1871661", session=session, retries=2, backoff=0.0)
    assert res_200["status_code"] == 200
    assert res_200["fsa_status"] == "active"
    assert res_200["businessname"] == "Taste Of Kerala"
    assert res_200["ratingvalue"] == "5"
    assert res_200["newratingpending"] is False
    assert abs(res_200["latitude"] - 51.3847167) < 1e-6

    # 2. 404 delisted response
    session.get.return_value = _mock_response(404)
    res_404 = probe_establishment("35993", session=session, retries=2, backoff=0.0)
    assert res_404["status_code"] == 404
    assert res_404["fsa_status"] == "delisted"

    # 3. 503 transient failure -> retries then returns fsa_status=None (never marks delisted)
    session.get.return_value = _mock_response(503)
    res_503 = probe_establishment("999999", session=session, retries=3, backoff=0.0)
    assert res_503["status_code"] == 503
    assert res_503["fsa_status"] is None
    assert res_503["error"] == "HTTP 503"

    # 4. Network exception on attempt 1 followed by 200 on attempt 2
    session.get.side_effect = [
        requests.exceptions.Timeout("timed out"),
        _mock_response(200, {"FHRSID": 123, "BusinessName": "Recovered Cafe", "RatingValue": "4"}),
    ]
    res_recovered = probe_establishment("123", session=session, retries=2, backoff=0.0)
    assert res_recovered["fsa_status"] == "active"
    assert res_recovered["businessname"] == "Recovered Cafe"


def test_both_live_pairs_are_never_linked():
    rows = [
        {
            "fhrsid": "1955974",
            "businessname": "Peabodys Coffee",
            "postcode": "SW17 0QT",
            "addressline1": "Lanesborough Wing St Georges Hospital",
            "first_seen": "2026-06-08",
            "latitude": 51.4267,
            "longitude": -0.1757,
            "user_rating": None,
            "in_scope": False,
        },
        {
            "fhrsid": "1955975",
            "businessname": "Peabodys Coffee",
            "postcode": "SW17 0QT",
            "addressline1": "Atkinson Morley Wing St Georges Hospital",
            "first_seen": "2026-06-08",
            "latitude": 51.4267,
            "longitude": -0.1757,
            "user_rating": None,
            "in_scope": False,
        },
    ]
    probes = {
        "1955974": {"fhrsid": "1955974", "status_code": 200, "fsa_status": "active"},
        "1955975": {"fhrsid": "1955975", "status_code": 200, "fsa_status": "active"},
    }
    out = {r["fhrsid"]: r for r in reconcile_in_memory(rows, probes, today="2026-10-12")}
    assert out["1955974"]["fsa_status"] == "active"
    assert out["1955975"]["fsa_status"] == "active"
    assert out["1955974"]["superseded_by"] is None
    assert out["1955975"]["superseded_by"] is None


def test_relisted_pair_carries_rating_and_respects_null_only_for_metadata():
    rows = [
        {
            "fhrsid": "1944601",
            "businessname": "Coppia Coffee",
            "postcode": "SW10 9NZ",
            "first_seen": "2026-05-18",
            "latitude": 51.4883,
            "longitude": -0.1782,
            "user_rating": 1,
            "rating_source": "desk",
            "user_rating_10pt_backup": 1,
            "in_scope": True,
            "gemini_insights_structured": '{"old": true}',
            "match_score": 42,
            "maps_rating": 4.4,
        },
        {
            "fhrsid": "1956432",
            "businessname": "  coppia coffee ",
            "postcode": "sw10 9nz",
            "first_seen": "2026-06-08",
            "latitude": None,
            "longitude": None,
            "user_rating": None,
            "rating_source": None,
            "user_rating_10pt_backup": None,
            "in_scope": False,
            "gemini_insights_structured": '{"new": true}',
            "match_score": 78,
            "maps_rating": None,
        },
    ]
    probes = {
        "1944601": {"fhrsid": "1944601", "status_code": 404, "fsa_status": "delisted"},
        "1956432": {"fhrsid": "1956432", "status_code": 200, "fsa_status": "active"},
    }
    out = {r["fhrsid"]: r for r in reconcile_in_memory(rows, probes, today="2026-10-12")}

    # Old row is retired and linked to replacement; its rating is preserved for BQML training
    assert out["1944601"]["fsa_status"] == "delisted"
    assert out["1944601"]["delisted_at"] == "2026-10-12"
    assert out["1944601"]["superseded_by"] == "1956432"
    assert out["1944601"]["user_rating"] == 1

    # Active replacement receives user_rating, rating_source, backup, maps_rating (where NULL),
    # and keeps its existing non-NULL gemini_insights_structured & match_score
    assert out["1956432"]["fsa_status"] == "active"
    assert out["1956432"]["superseded_by"] is None
    assert out["1956432"]["user_rating"] == 1
    assert out["1956432"]["rating_source"] == "desk"
    assert out["1956432"]["in_scope"] is True
    assert out["1956432"]["user_rating_10pt_backup"] == 1
    assert out["1956432"]["maps_rating"] == 4.4
    assert out["1956432"]["gemini_insights_structured"] == '{"new": true}'
    assert out["1956432"]["match_score"] == 78

    # Idempotent re-run preserves delisted_at and all carried fields
    second = {r["fhrsid"]: r for r in reconcile_in_memory(out.values(), probes, today="2026-10-19")}
    assert second["1944601"]["delisted_at"] == "2026-10-12"
    assert second["1944601"]["superseded_by"] == "1956432"
    assert second["1956432"]["user_rating"] == 1
    assert second["1956432"]["match_score"] == 78


def test_rating_conflict_newest_by_first_seen_wins():
    # Case A: Active row has newer first_seen -> active row's rating wins
    rows_a = [
        {
            "fhrsid": "100",
            "businessname": "Banh Banh",
            "postcode": "SE15 4BQ",
            "first_seen": "2025-11-04",
            "latitude": 51.469,
            "longitude": -0.068,
            "user_rating": 3,
            "rating_source": "desk",
            "in_scope": True,
        },
        {
            "fhrsid": "200",
            "businessname": "Banh Banh",
            "postcode": "SE15 4BQ",
            "first_seen": "2026-06-15",
            "latitude": 51.469,
            "longitude": -0.068,
            "user_rating": 5,
            "rating_source": "visited",
            "in_scope": True,
        },
    ]
    probes = {
        "100": {"fhrsid": "100", "status_code": 404, "fsa_status": "delisted"},
        "200": {"fhrsid": "200", "status_code": 200, "fsa_status": "active"},
    }
    out_a = {r["fhrsid"]: r for r in reconcile_in_memory(rows_a, probes, today="2026-10-12")}
    assert out_a["200"]["user_rating"] == 5
    assert out_a["200"]["rating_source"] == "visited"
    assert out_a["100"]["user_rating"] == 3  # Old value stays on retired row

    # Case B: Delisted row has newer first_seen -> delisted row's rating wins on active row,
    # and old value remains on delisted row
    rows_b = [
        {
            "fhrsid": "100",
            "businessname": "Banh Banh",
            "postcode": "SE15 4BQ",
            "first_seen": "2026-08-01",
            "latitude": 51.469,
            "longitude": -0.068,
            "user_rating": 6,
            "rating_source": "visited",
            "in_scope": True,
        },
        {
            "fhrsid": "200",
            "businessname": "Banh Banh",
            "postcode": "SE15 4BQ",
            "first_seen": "2026-01-10",
            "latitude": 51.469,
            "longitude": -0.068,
            "user_rating": 2,
            "rating_source": "desk",
            "in_scope": True,
        },
    ]
    out_b = {r["fhrsid"]: r for r in reconcile_in_memory(rows_b, probes, today="2026-10-12")}
    assert out_b["200"]["user_rating"] == 6
    assert out_b["200"]["rating_source"] == "visited"
    assert out_b["100"]["user_rating"] == 6


def test_distance_over_100m_is_not_linked():
    rows = [
        {
            "fhrsid": "1",
            "businessname": "Chain Cafe",
            "postcode": "SW1A 1AA",
            "first_seen": "2025-06-13",
            "latitude": 51.5010,
            "longitude": -0.1415,
            "user_rating": 4,
            "in_scope": True,
        },
        {
            "fhrsid": "2",
            "businessname": "Chain Cafe",
            "postcode": "SW1A 1AA",
            "first_seen": "2026-06-13",
            "latitude": 51.5050,  # ~445 m north
            "longitude": -0.1415,
            "user_rating": None,
            "in_scope": True,
        },
    ]
    probes = {
        "1": {"fhrsid": "1", "status_code": 404, "fsa_status": "delisted"},
        "2": {"fhrsid": "2", "status_code": 200, "fsa_status": "active"},
    }
    out = {r["fhrsid"]: r for r in reconcile_in_memory(rows, probes, today="2026-10-12")}
    assert out["1"]["superseded_by"] is None
    assert out["2"]["user_rating"] is None


def test_sql_builders_enforce_liveness_and_100m_rules():
    merge_sql = build_liveness_merge_sql("proj.ds.fsa_master", "proj.ds.fsa_liveness_check", "2026-10-12")
    assert "fsa_status = 'delisted'" in merge_sql
    assert "last_seen = DATE('2026-10-12')" in merge_sql

    link_sql = build_link_superseded_sql("proj.ds.fsa_master", new_ids=["1956432"])
    assert "D.fsa_status = 'delisted'" in link_sql
    assert "A.fsa_status = 'active'" in link_sql
    assert "<= 100.0" in link_sql
    assert "'1956432'" in link_sql

    carry_sql = build_carry_over_sql("proj.ds.fsa_master", new_ids=["1956432"])
    assert "S.d_rated_first_seen > T.first_seen" in carry_sql
    assert "COALESCE(T.user_rating_10pt_backup, S.d_user_rating_10pt_backup)" in carry_sql

    non_std_sql = build_candidates_query("proj.ds.fsa_master", only_non_standard_ratings=True)
    assert "fsa_status IS DISTINCT FROM 'delisted'" in non_std_sql
    assert "ratingvalue NOT IN ('1', '2', '3', '4', '5')" in non_std_sql
