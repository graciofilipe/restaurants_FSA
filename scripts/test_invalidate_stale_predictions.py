"""Tests for clearing predictions left behind by a superseded model.

The statement itself is one `UPDATE`. What these tests protect is the decision
around it: that "stale" is defined by the served model's own timestamp rather
than a date somebody typed, that nothing is written without `--execute`, and
that a missing model stops the run instead of clearing the whole column.
"""
import datetime
import unittest
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound

from scripts.invalidate_stale_predictions import (
    build_impact_query,
    build_invalidate_sql,
    read_model_trained_at,
    run_invalidation,
)

TABLE = "proj.ds.fsa_master"
TRAINED_AT = datetime.datetime(2026, 9, 23, 15, 5, 0, tzinfo=datetime.timezone.utc)


class TestStalenessIsTheModelsOwnTimestamp(unittest.TestCase):
    """A prediction is stale if the model that made it has been replaced."""

    def test_the_cutoff_comes_from_the_served_model(self):
        client = MagicMock()
        client.get_model.return_value = MagicMock(created=TRAINED_AT, modified=TRAINED_AT)

        self.assertEqual(
            read_model_trained_at(client, "proj", "ds", "restaurant_preference_model"),
            TRAINED_AT)
        client.get_model.assert_called_once_with("proj.ds.restaurant_preference_model")

    def test_a_metadata_edit_does_not_move_the_cutoff(self):
        """`created` is reset by CREATE OR REPLACE MODEL; `modified` also moves
        for a description or label change, which would clear good rows.
        """
        later = TRAINED_AT + datetime.timedelta(days=3)
        client = MagicMock()
        client.get_model.return_value = MagicMock(created=TRAINED_AT, modified=later)

        self.assertEqual(
            read_model_trained_at(client, "proj", "ds", "restaurant_preference_model"),
            TRAINED_AT)

    def test_the_cutoff_is_interpolated_into_the_update(self):
        sql = build_invalidate_sql(TABLE, TRAINED_AT)
        self.assertIn("2026-09-23 15:05:00+00:00", sql)
        self.assertIn(f"UPDATE `{TABLE}`", sql)

    def test_both_the_score_and_its_stamp_are_cleared(self):
        sql = build_invalidate_sql(TABLE, TRAINED_AT)
        self.assertIn("predicted_user_rating = NULL", sql)
        self.assertIn("predicted_at = NULL", sql)

    def test_a_prediction_from_the_current_model_is_left_alone(self):
        """The predicate must be `<`, not `IS NOT NULL`.

        Clearing every prediction would be a far more expensive mistake than it
        looks: the queue re-scores them, and any row without a Gemini profile
        bills a fresh `AI.GENERATE` on the way through.
        """
        sql = build_invalidate_sql(TABLE, TRAINED_AT)
        self.assertIn("predicted_at <", sql)
        self.assertNotIn("predicted_at IS NOT NULL", sql)

    def test_a_row_that_was_never_scored_is_not_touched(self):
        """`predicted_at IS NULL` must not satisfy the predicate.

        A NULL comparison is already false in SQL, so this is really a guard
        against somebody later adding an `OR predicted_at IS NULL` for tidiness.
        """
        sql = build_invalidate_sql(TABLE, TRAINED_AT)
        self.assertNotIn("IS NULL", sql.split("WHERE", 1)[1])


class TestImpactIsMeasuredBeforeTheWrite(unittest.TestCase):
    """The row count alone does not describe the cost of re-scoring."""

    def test_the_impact_query_separates_profiled_from_unprofiled(self):
        sql = build_impact_query(TABLE, TRAINED_AT)
        self.assertIn("gemini_profiled_at IS NULL", sql)
        self.assertIn("gemini_profiled_at IS NOT NULL", sql)

    def test_the_impact_query_reads_the_same_cutoff_as_the_update(self):
        self.assertIn("2026-09-23 15:05:00+00:00", build_impact_query(TABLE, TRAINED_AT))


class TestNothingRunsWithoutExecute(unittest.TestCase):

    def _client_returning(self, trained_at, impact):
        client = MagicMock()
        client.get_model.return_value = MagicMock(created=trained_at, modified=trained_at)
        impact_row = MagicMock()
        impact_row.items.return_value = impact.items()

        def query(sql, **kwargs):
            job = MagicMock()
            job.result.return_value = [impact_row]
            job.num_dml_affected_rows = impact.get("stale_predictions", 0)
            job.total_bytes_processed = 5_000_000
            return job

        client.query.side_effect = query
        return client

    @patch("scripts.invalidate_stale_predictions.bigquery.Client")
    def test_a_dry_run_writes_nothing(self, mock_client_cls):
        """The UPDATE is still submitted -- that is how BigQuery validates it --
        but it must carry `dry_run=True`, which is what makes it free and inert.
        """
        client = self._client_returning(
            TRAINED_AT, {"stale_predictions": 1065, "stale_and_unprofiled": 43})
        mock_client_cls.return_value = client

        run_invalidation(bq_path=TABLE, execute=False)

        updates = [call for call in client.query.call_args_list
                   if call.args[0].lstrip().startswith("UPDATE")]
        self.assertEqual(len(updates), 1)
        self.assertTrue(updates[0].kwargs["job_config"].dry_run)

    @patch("scripts.invalidate_stale_predictions.bigquery.Client")
    def test_execute_issues_exactly_one_update(self, mock_client_cls):
        client = self._client_returning(
            TRAINED_AT, {"stale_predictions": 1065, "stale_and_unprofiled": 43})
        mock_client_cls.return_value = client

        run_invalidation(bq_path=TABLE, execute=True)

        updates = [call.args[0] for call in client.query.call_args_list
                   if call.args[0].lstrip().startswith("UPDATE")]
        self.assertEqual(len(updates), 1)
        self.assertIn("2026-09-23 15:05:00+00:00", updates[0])

    @patch("scripts.invalidate_stale_predictions.bigquery.Client")
    def test_a_missing_model_stops_the_run(self, mock_client_cls):
        """No model, no cutoff. Guessing one would clear the column wholesale."""
        client = MagicMock()
        client.get_model.side_effect = NotFound("no such model")
        mock_client_cls.return_value = client

        with self.assertRaises(RuntimeError):
            run_invalidation(bq_path=TABLE, execute=True)

        self.assertFalse([call.args[0] for call in client.query.call_args_list
                          if call.args[0].lstrip().startswith("UPDATE")])


if __name__ == "__main__":
    unittest.main()
