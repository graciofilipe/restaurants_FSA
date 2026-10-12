"""Unit tests for `scripts/migrate_fsa_status_columns.py`."""

from app.services.bq_utils import MASTER_BQ_SCHEMA
from scripts.migrate_fsa_status_columns import (
    FSA_STATUS_COLUMNS,
    build_add_columns_sql,
    build_backfill_sql,
)


def test_status_columns_present_in_master_bq_schema():
    schema_map = {field.name: field.field_type for field in MASTER_BQ_SCHEMA}
    for col_name, col_type in FSA_STATUS_COLUMNS:
        assert col_name in schema_map, f"Missing {col_name} in MASTER_BQ_SCHEMA"
        assert schema_map[col_name] == col_type


def test_migration_sql_builders():
    add_sql = build_add_columns_sql("proj.ds.fsa_master")
    assert "ADD COLUMN IF NOT EXISTS last_seen DATE" in add_sql
    assert "ADD COLUMN IF NOT EXISTS fsa_status STRING" in add_sql
    assert "ADD COLUMN IF NOT EXISTS delisted_at DATE" in add_sql
    assert "ADD COLUMN IF NOT EXISTS superseded_by STRING" in add_sql

    backfill_sql = build_backfill_sql("proj.ds.fsa_master")
    assert "COALESCE(fsa_status, 'active')" in backfill_sql
    assert "COALESCE(last_seen, first_seen)" in backfill_sql
