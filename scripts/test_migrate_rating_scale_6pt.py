"""Unit tests for `scripts/migrate_rating_scale_6pt.py`."""
from unittest.mock import MagicMock, patch

import pytest

from scripts.migrate_rating_scale_6pt import (
    build_add_7pt_backup_column_sql,
    build_add_backup_column_sql,
    build_backfill_7pt_backup_sql,
    build_convert_rating_sql,
    convert_7pt_to_6pt,
    convert_10pt_to_6pt,
    run_migration,
)


def test_convert_10pt_to_6pt_exact_mapping():
    expected = {
        1: 1,
        2: 2,
        3: 2,
        4: 3,
        5: 3,
        6: 4,
        7: 4,
        8: 5,
        9: 5,
        10: 6,
    }
    for s10, s6 in expected.items():
        assert convert_10pt_to_6pt(s10) == s6

    with pytest.raises(ValueError):
        convert_10pt_to_6pt(0)
    with pytest.raises(ValueError):
        convert_10pt_to_6pt(11)


def test_convert_7pt_to_6pt_exact_mapping():
    expected = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 5,
        7: 6,
    }
    for s7, s6 in expected.items():
        assert convert_7pt_to_6pt(s7) == s6

    with pytest.raises(ValueError):
        convert_7pt_to_6pt(0)
    with pytest.raises(ValueError):
        convert_7pt_to_6pt(8)


def test_sql_builders_preserve_backups_and_apply_idempotent_affine_formulas():
    add_10_sql = build_add_backup_column_sql("p.d.t")
    assert "ALTER TABLE `p.d.t`" in add_10_sql
    assert "ADD COLUMN IF NOT EXISTS user_rating_10pt_backup INT64" in add_10_sql

    add_7_sql = build_add_7pt_backup_column_sql("p.d.t")
    assert "ALTER TABLE `p.d.t`" in add_7_sql
    assert "ADD COLUMN IF NOT EXISTS user_rating_7pt_backup INT64" in add_7_sql

    backfill_sql = build_backfill_7pt_backup_sql("p.d.t")
    assert "SET user_rating_7pt_backup = user_rating" in backfill_sql
    assert "user_rating_10pt_backup IS NULL" in backfill_sql
    assert "user_rating_7pt_backup IS NULL" in backfill_sql

    convert_sql = build_convert_rating_sql("p.d.t")
    assert "CAST(ROUND(1 + (user_rating_10pt_backup - 1) * 5.0 / 9.0) AS INT64)" in convert_sql
    assert "CAST(ROUND(1 + (user_rating_7pt_backup - 1) * 5.0 / 6.0) AS INT64)" in convert_sql


@patch("scripts.migrate_rating_scale_6pt.bigquery.Client")
def test_dry_run_submits_only_dry_run_query(mock_bq):
    client = MagicMock()
    job = MagicMock()
    job.total_bytes_processed = 2048
    client.query.return_value = job
    mock_bq.return_value = client

    out = run_migration("p.d.t", execute=False)

    assert out == {"dry_run": True, "bytes_processed": 2048}
    assert client.query.call_count == 1
    assert client.query.call_args.kwargs["job_config"].dry_run is True


@patch("scripts.migrate_rating_scale_6pt.bigquery.Client")
def test_execute_runs_backup_and_conversion_and_checks_invariants(mock_bq):
    before_row = {
        "total_rows": 11268,
        "labeled_rows": 518,
        "min_user_rating": 1,
        "max_user_rating": 6,
        "rows_above_6": 0,
        "visited_count": 46,
        "desk_count": 460,
        "null_source_count": 12,
    }
    after_row = {
        "total_rows": 11268,
        "labeled_rows": 518,
        "min_user_rating": 1,
        "max_user_rating": 5,
        "rows_above_6": 0,
        "visited_count": 46,
        "desk_count": 460,
        "null_source_count": 12,
    }
    before_job = MagicMock()
    before_job.result.return_value = [before_row]
    ddl_10_job = MagicMock()
    ddl_7_job = MagicMock()
    backfill_job = MagicMock()
    backfill_job.num_dml_affected_rows = 1
    convert_job = MagicMock()
    convert_job.num_dml_affected_rows = 518
    after_job = MagicMock()
    after_job.result.return_value = [after_row]

    client = MagicMock()
    client.query.side_effect = [
        before_job,
        ddl_10_job,
        ddl_7_job,
        backfill_job,
        convert_job,
        after_job,
    ]
    mock_bq.return_value = client

    out = run_migration("p.d.t", execute=True)

    assert out["dry_run"] is False
    assert out["backed_up_7pt_rows"] == 1
    assert out["converted_rows"] == 518
    assert out["after"]["max_user_rating"] == 5
    assert out["after"]["rows_above_6"] == 0
