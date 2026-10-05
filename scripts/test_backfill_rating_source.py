"""Unit tests for `scripts/backfill_rating_source.py`."""
from unittest.mock import MagicMock, patch

import pytest

from scripts.backfill_rating_source import (
    build_backfill_where_clause,
    build_summary_sql,
    build_update_sql,
    run_backfill,
)


def test_backfill_where_clause_targets_only_null_source_obvious_negatives():
    clause = build_backfill_where_clause()
    assert "user_rating IS NOT NULL" in clause
    assert "rating_source IS NULL" in clause
    assert "COALESCE(in_scope, TRUE) IS FALSE" in clause
    assert "COALESCE(pillar_is_sit_down, TRUE) IS FALSE" in clause
    assert "COALESCE(pillar_establishment_type, 'RESTAURANT_DINING') != 'RESTAURANT_DINING'" in clause
    assert "user_rating <= 3" in clause


def test_update_sql_sets_rating_source_to_desk_only():
    sql = build_update_sql("p.d.t")
    assert "UPDATE `p.d.t`" in sql
    assert "SET rating_source = 'desk'" in sql
    assert "user_rating =" not in sql.split("WHERE")[0]


@patch("scripts.backfill_rating_source.bigquery.Client")
def test_dry_run_submits_only_dry_run_queries(mock_bq):
    client = MagicMock()
    job = MagicMock()
    job.total_bytes_processed = 4096
    client.query.return_value = job
    mock_bq.return_value = client

    out = run_backfill("p.d.t", execute=False)

    assert out == {"dry_run": True, "bytes_processed": 4096}
    assert client.query.call_count == 2
    for call in client.query.call_args_list:
        assert call.kwargs["job_config"].dry_run is True


@patch("scripts.backfill_rating_source.bigquery.Client")
def test_execute_runs_update_and_verifies_label_invariants(mock_bq):
    before_row = {
        "total_rows": 6562,
        "labeled_rows": 411,
        "sum_user_rating": 1004,
        "visited_count": 20,
        "desk_count": 14,
        "null_source_count": 377,
        "eligible_for_desk_backfill": 357,
    }
    after_row = {
        "total_rows": 6562,
        "labeled_rows": 411,
        "sum_user_rating": 1004,
        "visited_count": 20,
        "desk_count": 371,
        "null_source_count": 20,
        "eligible_for_desk_backfill": 0,
    }
    before_job = MagicMock()
    before_job.result.return_value = [before_row]
    update_job = MagicMock()
    update_job.num_dml_affected_rows = 357
    after_job = MagicMock()
    after_job.result.return_value = [after_row]

    client = MagicMock()
    client.query.side_effect = [before_job, update_job, after_job]
    mock_bq.return_value = client

    out = run_backfill("p.d.t", execute=True)

    assert out["dry_run"] is False
    assert out["updated_rows"] == 357
    assert out["after"]["null_source_count"] == 20
