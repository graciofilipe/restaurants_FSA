"""Tests for the additive pillar-column migration.

The migration itself is trivial DDL. What these tests protect is everything
around it: that the column list is generated rather than retyped, that a failed
`ALTER TABLE` is loud, that nothing executes without `--execute`, and that the
before/after check cannot be fooled by the very columns the migration adds.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.core.pillar_schema import NON_JSON_COLUMNS, PILLAR_FIELDS
from app.services.bq_utils import MASTER_BQ_SCHEMA
from scripts.migrate_pillar_columns import (
    NEW_COLUMNS,
    build_ddl_statements,
    build_fingerprint_query,
    run_migration,
)

TABLE = "proj.ds.fsa_master"


class TestColumnList(unittest.TestCase):
    """The column list must come from the canonical schema, not a second copy.

    A hand-retyped list here is exactly the fifth copy of the pillar schema that
    Phase 3 existed to prevent.
    """

    def test_every_canonical_field_gets_a_column(self):
        names = [name for name, _ in NEW_COLUMNS]
        for field in PILLAR_FIELDS:
            self.assertIn(field.column, names)
        for column, _ in NON_JSON_COLUMNS:
            self.assertIn(column, names)

    def test_the_column_count_is_the_schema_count(self):
        self.assertEqual(len(NEW_COLUMNS), len(PILLAR_FIELDS) + len(NON_JSON_COLUMNS))

    def test_types_match_the_canonical_schema(self):
        types = dict(NEW_COLUMNS)
        for field in PILLAR_FIELDS:
            self.assertEqual(types[field.column], field.bq_type, field.column)

    def test_the_enum_pillars_are_added_as_strings(self):
        """CAST(specificity_level AS INT64) is what silently zeroes pillar 4
        today. Adding the column as INT64 would bake that mistake into the
        table, where it is far more expensive to undo than in a query."""
        types = dict(NEW_COLUMNS)
        self.assertEqual(types['pillar_geo_specificity'], 'STRING')
        self.assertEqual(types['pillar_establishment_type'], 'STRING')

    def test_no_new_column_collides_with_an_existing_one(self):
        existing = {f.name for f in MASTER_BQ_SCHEMA}
        for name, _ in NEW_COLUMNS:
            self.assertNotIn(name, existing, f"{name} already exists on the table")


class TestDdl(unittest.TestCase):

    def test_every_statement_is_add_column_if_not_exists(self):
        """Idempotence is what lets the migration run against the snapshot, then
        production, then again after a failure, without special-casing."""
        for stmt in build_ddl_statements(TABLE):
            self.assertIn("ADD COLUMN IF NOT EXISTS", stmt)

    def test_the_ddl_only_adds(self):
        joined = " ".join(build_ddl_statements(TABLE)).upper()
        for keyword in ('DROP', 'RENAME', 'SET OPTIONS', 'DELETE', 'UPDATE', 'INSERT'):
            self.assertNotIn(keyword, joined)

    def test_one_statement_per_column(self):
        self.assertEqual(len(build_ddl_statements(TABLE)), len(NEW_COLUMNS))

    def test_the_target_table_is_the_one_passed_in(self):
        """The default is production. A test that silently hit the default would
        be a test that migrates the live table."""
        for stmt in build_ddl_statements("proj.ds.snapshot"):
            self.assertIn("`proj.ds.snapshot`", stmt)
            self.assertNotIn("fsa_master", stmt)


class TestFingerprint(unittest.TestCase):
    """The before/after check.

    `ADD COLUMN` is metadata-only, so this should never fire -- which is the
    point: it is cheap, and it is the only evidence that the irreplaceable
    hand-entered labels came through untouched.
    """

    def test_the_fingerprint_covers_only_pre_existing_columns(self):
        """Fingerprinting `TO_JSON_STRING(t)` would include the newly added
        NULL columns and change between the two readings for no reason,
        making the check useless precisely when it is needed."""
        sql = build_fingerprint_query(TABLE)
        for field in MASTER_BQ_SCHEMA:
            self.assertIn(field.name, sql)
        for name, _ in NEW_COLUMNS:
            self.assertNotIn(name, sql)

    def test_the_fingerprint_counts_the_labels_explicitly(self):
        """411 hand-entered ratings are the one thing in this table that cannot
        be regenerated, so they get their own counter rather than being trusted
        to the hash."""
        self.assertIn("COUNT(user_rating)", build_fingerprint_query(TABLE))

    def test_the_fingerprint_query_only_reads(self):
        sql = build_fingerprint_query(TABLE).upper()
        for keyword in ('INSERT', 'UPDATE', 'DELETE', 'MERGE', 'CREATE', 'DROP', 'ALTER'):
            self.assertNotIn(keyword, sql)


class TestRunMigration(unittest.TestCase):

    def setUp(self):
        patcher = patch('scripts.migrate_pillar_columns.bigquery.Client')
        self.client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.client_cls.return_value
        # A stable fingerprint by default, so only the tests that care about it
        # have to say anything about it.
        self.client.query.return_value.result.return_value = [
            MagicMock(row_count=11268, labels=411, fingerprint=12345)
        ]

    def _executed(self):
        return [call.args[0] for call in self.client.query.call_args_list]

    def test_dry_run_is_the_default_and_nothing_reaches_bigquery_for_real(self):
        """The DDL text *is* submitted -- that is how it gets validated -- so
        the property that matters is that every submission carries dry_run."""
        run_migration(TABLE)
        self.assertTrue(self.client.query.called)
        for call in self.client.query.call_args_list:
            config = call.kwargs.get('job_config')
            self.assertIsNotNone(config, f"submitted without a job_config: {call.args[0]}")
            self.assertTrue(config.dry_run, f"submitted for real: {call.args[0]}")

    def test_dry_run_still_validates_against_bigquery(self):
        """Printing SQL nobody parsed is not a dry run. A typo in a type name
        should surface before the approval, not after it."""
        run_migration(TABLE)
        configs = [call.kwargs.get('job_config') for call in self.client.query.call_args_list]
        self.assertTrue(configs, "dry run issued no query at all")
        self.assertTrue(all(c is not None and c.dry_run for c in configs))

    def test_execute_issues_every_ddl_statement(self):
        run_migration(TABLE, execute=True)
        issued = [s for s in self._executed() if "ALTER TABLE" in s.upper()]
        self.assertEqual(len(issued), len(NEW_COLUMNS))

    def test_execute_fingerprints_before_and_after(self):
        run_migration(TABLE, execute=True)
        fingerprints = [s for s in self._executed() if "COUNT(user_rating)" in s]
        self.assertEqual(len(fingerprints), 2)

    def test_a_failed_ddl_raises_instead_of_logging_a_notice(self):
        """`migrate_to_in_scope_workflow.py` downgrades DDL failures to a
        warning. Here a half-applied schema would let Phase 5's backfill run
        against columns that do not exist, so it has to stop."""
        self.client.query.side_effect = RuntimeError("quota exceeded")
        with self.assertRaises(RuntimeError):
            run_migration(TABLE, execute=True)

    def test_a_changed_fingerprint_is_reported_as_a_failure(self):
        rows = [
            MagicMock(row_count=11268, labels=411, fingerprint=1),
            MagicMock(row_count=11268, labels=410, fingerprint=2),
        ]
        self.client.query.return_value.result.side_effect = (
            lambda *a, **k: [rows.pop(0)] if rows else [MagicMock(row_count=0, labels=0, fingerprint=0)]
        )
        with self.assertRaises(RuntimeError):
            run_migration(TABLE, execute=True)


if __name__ == '__main__':
    unittest.main()
