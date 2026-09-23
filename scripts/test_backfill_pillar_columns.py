"""Tests for the Phase 5 backfill.

This is the first phase that changes data you can see, so most of these tests
are about blast radius rather than correctness of the SQL: what each statement
is allowed to touch, what it must leave alone, and what order they run in.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.core.pillar_schema import PILLAR_FIELDS
from scripts.backfill_pillar_columns import (
    MAPS_BACKFILL_LOOKUP_AT,
    build_all_statements,
    build_in_scope_rederivation,
    build_maps_hit_backfill,
    build_maps_miss_backfill,
    build_pillar_backfill,
    build_profiled_at_backfill,
    build_validation_query,
    run_backfill,
)

TABLE = "proj.ds.fsa_master"


class TestPillarBackfill(unittest.TestCase):

    def test_every_canonical_column_is_written(self):
        sql = build_pillar_backfill(TABLE)
        for field in PILLAR_FIELDS:
            self.assertIn(f"{field.column} =", sql, field.column)

    def test_it_reads_the_nested_paths(self):
        """The regression guard for D2, at the point where the wrong path would
        be written into a column instead of merely computed in a query."""
        sql = build_pillar_backfill(TABLE)
        self.assertIn("$.1_value_and_volume.rating", sql)
        self.assertNotIn("1_value_and_volume_rating", sql)

    def test_missing_values_are_not_defaulted(self):
        """A profile that omits a score must leave NULL. IFNULL(..., 0) here
        would write the D2 failure permanently into the table, where no later
        query could tell it from a real zero."""
        sql = build_pillar_backfill(TABLE)
        self.assertNotIn("IFNULL", sql)
        self.assertNotIn("COALESCE", sql)

    def test_it_only_touches_profiled_rows(self):
        """8,501 rows have no profile. Writing NULLs over them would be a
        no-op that still rewrites the whole table."""
        self.assertIn("gemini_insights_structured IS NOT NULL", build_pillar_backfill(TABLE))

    def test_typed_columns_use_safe_cast(self):
        """One profile returning "N/A" for a score must not fail the statement
        for the other 2,765."""
        self.assertIn("SAFE_CAST", build_pillar_backfill(TABLE))


class TestProfiledAtBackfill(unittest.TestCase):

    def test_existing_profiles_count_as_fresh(self):
        """D-03: NULLing this would schedule a re-profile of all 2,767 rows,
        in a track whose headline defect is paying twice for Gemini."""
        sql = build_profiled_at_backfill(TABLE)
        self.assertIn("CURRENT_TIMESTAMP()", sql)
        self.assertIn("gemini_insights_structured IS NOT NULL", sql)

    def test_it_does_not_refresh_a_timestamp_that_already_exists(self):
        """Idempotence. Re-running must not slide the staleness clock forward
        and quietly postpone every future refresh."""
        self.assertIn("gemini_profiled_at IS NULL", build_profiled_at_backfill(TABLE))


class TestMapsSentinel(unittest.TestCase):

    def test_a_miss_is_recorded_as_looked_up_and_not_found(self):
        sql = build_maps_miss_backfill(TABLE)
        self.assertIn("maps_found = FALSE", sql)
        self.assertIn("maps_rating = NULL", sql)
        self.assertIn("maps_reviews = NULL", sql)
        self.assertIn("maps_rating = -1", sql)

    def test_the_miss_timestamp_is_in_the_past(self):
        """It takes over the do-not-retry role from -1, so it must be non-NULL.
        Past rather than now because we do not know when the lookup happened,
        and a future staleness sweep should re-check rather than trust it."""
        self.assertIn("2026-01-01", MAPS_BACKFILL_LOOKUP_AT)
        self.assertIn(MAPS_BACKFILL_LOOKUP_AT, build_maps_miss_backfill(TABLE))

    def test_a_hit_is_recorded_as_found_without_touching_the_rating(self):
        sql = build_maps_hit_backfill(TABLE)
        self.assertIn("maps_found = TRUE", sql)
        self.assertNotIn("maps_rating =", sql)

    def test_a_hit_is_anything_with_a_real_rating(self):
        sql = build_maps_hit_backfill(TABLE)
        self.assertIn("maps_rating > 0", sql)

    def test_never_looked_up_rows_are_left_alone(self):
        """8,762 rows have no Maps lookup. maps_lookup_at must stay NULL for
        them, or the Phase 6 guard reads them as already tried and they become
        permanently ineligible for enrichment."""
        # The miss statement's sentinel equality already excludes NULL; the hit
        # statement needs the guard spelled out, since `NULL > 0` is not FALSE.
        self.assertIn("maps_rating = -1", build_maps_miss_backfill(TABLE))
        self.assertIn("maps_rating IS NOT NULL", build_maps_hit_backfill(TABLE))


class TestInScopeRederivation(unittest.TestCase):
    """D13. The largest behavioural change in the track."""

    def test_it_derives_from_the_typed_column(self):
        self.assertIn("in_scope = pillar_is_sit_down", build_in_scope_rederivation(TABLE))

    def test_it_never_overrules_a_human(self):
        """214 of the 369 trainable labels sit on rows the profiler calls
        not-sit-down. Re-deriving those would overwrite a human triage decision
        with a model's opinion and collapse the training set to 157."""
        sql = build_in_scope_rederivation(TABLE)
        self.assertIn("user_rating IS NULL", sql)
        self.assertIn("rating_source IS NULL", sql)

    def test_it_skips_rows_with_nothing_to_derive_from(self):
        """8,501 rows were never profiled, and one profile is unparseable."""
        self.assertIn("pillar_is_sit_down IS NOT NULL", build_in_scope_rederivation(TABLE))

    def test_it_only_writes_where_the_value_would_change(self):
        sql = build_in_scope_rederivation(TABLE)
        self.assertIn("IS DISTINCT FROM", sql)

    def test_it_touches_only_in_scope(self):
        """It must not become a general-purpose fixer. in_scope governs spend;
        anything else changing here would be unreviewed."""
        body = build_in_scope_rederivation(TABLE).split("WHERE")[0]
        self.assertEqual(body.count("="), 1)


class TestStatementOrdering(unittest.TestCase):

    def test_in_scope_is_derived_after_the_column_it_reads_is_filled(self):
        """`in_scope = pillar_is_sit_down` against an unfilled column would set
        every row's in_scope to NULL."""
        labels = [label for label, _, _ in build_all_statements(TABLE)]
        self.assertLess(labels.index('pillars'), labels.index('in_scope'))

    def test_the_maps_statements_run_before_the_sentinel_is_erased(self):
        """Both read `maps_rating = -1` / `> 0`, and the miss statement erases
        the sentinel it selects on. Reversing them would mark misses as hits."""
        labels = [label for label, _, _ in build_all_statements(TABLE)]
        self.assertLess(labels.index('maps_hit'), labels.index('maps_miss'))

    def test_every_statement_has_a_matching_count_query(self):
        """The dry run reports what each statement would touch. A statement
        without one would execute unpreviewed."""
        for label, update_sql, count_sql in build_all_statements(TABLE):
            self.assertTrue(count_sql.strip().upper().startswith("SELECT"), label)
            self.assertIn("COUNT(*)", count_sql, label)

    def test_no_statement_writes_to_another_table(self):
        for label, update_sql, _ in build_all_statements("proj.ds.snapshot"):
            self.assertIn("`proj.ds.snapshot`", update_sql, label)
            self.assertNotIn("fsa_master", update_sql, label)

    def test_every_statement_is_an_update(self):
        """Nothing here should delete, drop or replace. The one irreplaceable
        thing in this table is 411 hand-entered labels."""
        for label, update_sql, _ in build_all_statements(TABLE):
            self.assertTrue(update_sql.strip().upper().startswith("UPDATE"), label)
            for keyword in ('DELETE', 'DROP', 'TRUNCATE', 'CREATE OR REPLACE', 'INSERT'):
                self.assertNotIn(keyword, update_sql.upper(), label)


class TestValidation(unittest.TestCase):

    def test_it_reports_coverage_for_every_column(self):
        sql = build_validation_query(TABLE)
        for field in PILLAR_FIELDS:
            self.assertIn(field.column, sql)

    def test_it_counts_distinct_values_for_the_features(self):
        """A feature column with one distinct value is a dead feature, which is
        the exact condition this track exists to end. Coverage alone would not
        catch a column backfilled entirely to the same number."""
        sql = build_validation_query(TABLE)
        self.assertIn("COUNT(DISTINCT", sql)

    def test_it_only_reads(self):
        sql = build_validation_query(TABLE).upper()
        for keyword in ('UPDATE', 'DELETE', 'MERGE', 'DROP', 'ALTER', 'INSERT'):
            self.assertNotIn(keyword, sql)


class TestRunBackfill(unittest.TestCase):

    def setUp(self):
        patcher = patch('scripts.backfill_pillar_columns.bigquery.Client')
        self.client_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.client = self.client_cls.return_value
        self.client.query.return_value.result.return_value = [MagicMock(affected=0)]
        self.client.query.return_value.num_dml_affected_rows = 0

    def test_dry_run_is_the_default_and_writes_nothing(self):
        run_backfill(TABLE)
        for call in self.client.query.call_args_list:
            config = call.kwargs.get('job_config')
            sql = call.args[0]
            if sql.strip().upper().startswith("UPDATE"):
                self.assertIsNotNone(config, f"UPDATE submitted with no job_config: {sql[:60]}")
                self.assertTrue(config.dry_run, f"UPDATE submitted for real: {sql[:60]}")

    def test_dry_run_validates_every_update_against_bigquery(self):
        run_backfill(TABLE)
        validated = [c.args[0] for c in self.client.query.call_args_list
                     if c.args[0].strip().upper().startswith("UPDATE")]
        self.assertEqual(len(validated), len(build_all_statements(TABLE)))

    def test_execute_runs_every_statement(self):
        run_backfill(TABLE, execute=True)
        issued = [c.args[0] for c in self.client.query.call_args_list
                  if c.args[0].strip().upper().startswith("UPDATE")
                  and not (c.kwargs.get('job_config') and c.kwargs['job_config'].dry_run)]
        self.assertEqual(len(issued), len(build_all_statements(TABLE)))

    def test_a_failed_statement_stops_the_run(self):
        """The statements are ordered and dependent -- in_scope reads a column
        an earlier statement fills. Carrying on past a failure would derive
        in_scope from NULLs and take the whole table out of scope."""
        self.client.query.side_effect = RuntimeError("resources exceeded")
        with self.assertRaises(RuntimeError):
            run_backfill(TABLE, execute=True)


if __name__ == '__main__':
    unittest.main()
