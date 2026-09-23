"""Tests for retiring the V1 `gemini_insights` text column.

The column holds 1,116 free-text evaluations that exist nowhere else: the rows
carrying them have no structured V2 profile, so dropping the column without
archiving first destroys the only record. These tests exist to make that
impossible by accident -- the drop is gated on an archive that verifiably holds
every row, and nothing runs without `--execute`.
"""
import unittest
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound

from scripts.retire_v1_insights import (
    build_archive_sql,
    build_drop_sql,
    build_source_count_sql,
    run_retirement,
)

TABLE = "proj.ds.fsa_master"
ARCHIVE = "proj.ds.gemini_insights_v1_archive_20260923"


class TestTheArchiveKeepsTheText(unittest.TestCase):

    def test_the_text_and_enough_to_identify_it_are_kept(self):
        sql = build_archive_sql(TABLE, ARCHIVE)
        for column in ("fhrsid", "businessname", "postcode", "first_seen", "gemini_insights"):
            self.assertIn(column, sql)

    def test_only_rows_that_have_text_are_archived(self):
        self.assertIn("WHERE gemini_insights IS NOT NULL", build_archive_sql(TABLE, ARCHIVE))

    def test_an_existing_archive_is_not_overwritten(self):
        """Plain `CREATE TABLE`. `OR REPLACE` would let a re-run after the drop
        replace a good archive with an empty one.
        """
        sql = build_archive_sql(TABLE, ARCHIVE)
        self.assertIn(f"CREATE TABLE `{ARCHIVE}`", sql)
        self.assertNotIn("OR REPLACE", sql)
        self.assertNotIn("IF NOT EXISTS", sql)


class TestTheDropIsGatedOnTheArchive(unittest.TestCase):

    def _client(self, source_rows=1116, archive_rows=1116, archive_exists=True):
        client = MagicMock()
        if archive_exists:
            client.get_table.return_value = MagicMock(num_rows=archive_rows)
        else:
            client.get_table.side_effect = NotFound("no archive")

        def query(sql, **kwargs):
            job = MagicMock()
            row = MagicMock()
            row.rows_with_text = source_rows
            job.result.return_value = [row]
            job.total_bytes_processed = 1_500_000
            job.num_dml_affected_rows = source_rows
            return job

        client.query.side_effect = query
        return client

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_a_missing_archive_stops_the_drop(self, mock_client_cls):
        mock_client_cls.return_value = self._client(archive_exists=False)

        with self.assertRaises(RuntimeError):
            run_retirement(bq_path=TABLE, archive_path=ARCHIVE, drop=True, execute=True)

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_a_short_archive_stops_the_drop(self, mock_client_cls):
        """One row short is one evaluation lost. The count has to match exactly."""
        mock_client_cls.return_value = self._client(source_rows=1116, archive_rows=1115)

        with self.assertRaises(RuntimeError):
            run_retirement(bq_path=TABLE, archive_path=ARCHIVE, drop=True, execute=True)

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_a_complete_archive_lets_the_drop_through(self, mock_client_cls):
        client = self._client(source_rows=1116, archive_rows=1116)
        mock_client_cls.return_value = client

        run_retirement(bq_path=TABLE, archive_path=ARCHIVE, drop=True, execute=True)

        drops = [call.args[0] for call in client.query.call_args_list
                 if "DROP COLUMN" in call.args[0]]
        self.assertEqual(len(drops), 1)
        self.assertIn("gemini_insights", drops[0])

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_an_already_dropped_column_is_not_a_failure(self, mock_client_cls):
        """Re-running after a successful drop reports zero source rows. That is
        the finished state, not a missing archive.
        """
        client = self._client(source_rows=0, archive_rows=1116)
        mock_client_cls.return_value = client

        run_retirement(bq_path=TABLE, archive_path=ARCHIVE, drop=True, execute=True)


class TestNothingRunsWithoutExecute(unittest.TestCase):

    def _client(self):
        client = MagicMock()
        client.get_table.return_value = MagicMock(num_rows=1116)
        job = MagicMock()
        row = MagicMock()
        row.rows_with_text = 1116
        job.result.return_value = [row]
        job.total_bytes_processed = 1_500_000
        client.query.return_value = job
        return client

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_a_dry_run_creates_nothing(self, mock_client_cls):
        client = self._client()
        mock_client_cls.return_value = client

        run_retirement(bq_path=TABLE, archive_path=ARCHIVE, archive=True, execute=False)

        for call in client.query.call_args_list:
            if "CREATE TABLE" in call.args[0]:
                self.assertTrue(call.kwargs["job_config"].dry_run)

    @patch("scripts.retire_v1_insights.bigquery.Client")
    def test_a_dry_run_drops_nothing(self, mock_client_cls):
        client = self._client()
        mock_client_cls.return_value = client

        run_retirement(bq_path=TABLE, archive_path=ARCHIVE, drop=True, execute=False)

        executed = [call.args[0] for call in client.query.call_args_list
                    if "DROP COLUMN" in call.args[0]
                    and not call.kwargs.get("job_config", MagicMock()).dry_run]
        self.assertFalse(executed)


class TestTheCountQuery(unittest.TestCase):

    def test_it_counts_only_rows_carrying_text(self):
        sql = build_source_count_sql(TABLE)
        self.assertIn("COUNTIF(gemini_insights IS NOT NULL)", sql)
        self.assertIn(TABLE, sql)

    def test_the_drop_names_one_column(self):
        sql = build_drop_sql(TABLE)
        self.assertIn(f"ALTER TABLE `{TABLE}`", sql)
        self.assertIn("DROP COLUMN gemini_insights", sql)
        self.assertNotIn("gemini_insights_structured", sql)


if __name__ == "__main__":
    unittest.main()
