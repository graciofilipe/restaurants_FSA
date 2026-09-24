
import unittest
from unittest.mock import MagicMock, patch
from app.cron import fetch_weekly

class TestFetchWeekly(unittest.TestCase):

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_main_loop_iteration(self, mock_run_sync, mock_get_config):
        # Mock 2 config rows
        mock_get_config.return_value = [
            {'latitude': 51.5, 'longitude': -0.1, 'max_results': 100},
            {'latitude': 52.5, 'longitude': -0.2, 'max_results': 100}
        ]

        fetch_weekly.main()

        # Verify run_sync_for_config called twice
        self.assertEqual(mock_run_sync.call_count, 2)

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_main_loop_continues_on_error(self, mock_run_sync, mock_get_config):
        """One bad search area must not cost the others their run -- but see
        `TestTheJobReportsItsOwnFailure`, which pins that the job still exits
        non-zero afterwards."""
        mock_get_config.return_value = [
            {'latitude': 51.5, 'longitude': -0.1},
            {'latitude': 52.5, 'longitude': -0.2}
        ]

        # First call raises exception, second succeeds
        mock_run_sync.side_effect = [Exception("Test Error"), None]

        with self.assertRaises(SystemExit):
            fetch_weekly.main()

        # Verify run_sync_for_config called twice
        self.assertEqual(mock_run_sync.call_count, 2)


    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_run_sync_loads_only_fhrsids(self, mock_append, mock_process, mock_load, mock_fetch):
        """The cron only needs existing IDs to spot new restaurants."""
        mock_fetch.return_value = []
        mock_load.return_value = {'1', '2'}
        mock_process.return_value = ([], "Summary")

        fetch_weekly.run_sync_for_config({
            'latitude': 51.5074, 'longitude': -0.1278, 'max_results': 5000,
            'target_bq_table': 'p.d.t', 'radius': 5
        })

        mock_load.assert_called_once_with('p', 'd', 't')
        self.assertEqual(mock_process.call_args.args[0], {'1', '2'})

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_run_sync_aborts_when_ids_cannot_be_loaded(self, mock_append, mock_process, mock_load, mock_fetch):
        """Appending without knowing what already exists would duplicate the table.

        It aborts, and since D9 it also propagates: swallowing this left the
        job reporting success on a week where it ingested nothing.
        """
        mock_fetch.return_value = []
        mock_load.side_effect = Exception("credentials expired")

        with self.assertRaises(Exception):
            fetch_weekly.run_sync_for_config({
                'latitude': 51.5074, 'longitude': -0.1278, 'max_results': 5000,
                'target_bq_table': 'p.d.t', 'radius': 5
            })

        mock_process.assert_not_called()
        mock_append.assert_not_called()

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_run_sync_for_config_new_schema(self, mock_append, mock_process, mock_load, mock_fetch):
        # Config with new schema
        config = {
            'latitude': 51.5074, 
            'longitude': -0.1278, 
            'max_results': 5000, 
            'target_bq_table': 'p.d.t',
            'radius': 5
        }
        
        # Mock returns
        mock_fetch.return_value = [] # Return empty list to stop early or simple list
        mock_load.return_value = set()
        mock_process.return_value = ([], "Summary")
        
        fetch_weekly.run_sync_for_config(config)
        
        # Verify fetch_data_for_all_coordinates called with correct tuple
        # Expected: [(lon, lat)]
        mock_fetch.assert_called_with([(-0.1278, 51.5074)], 5000)



class TestTheJobReportsItsOwnFailure(unittest.TestCase):
    """D9, in the one place nobody is watching.

    `fetch_weekly` runs as a scheduled Cloud Run Job once a week. `main()`
    caught every exception, logged it, and returned -- so the process exited 0
    and Cloud Run recorded a success. A weekly ingest could fail on an expired
    credential for a month and the only evidence would be in logs nobody opens,
    plus a `first_seen` gap nobody would attribute to this.

    Exit status is the only signal Cloud Run reads, so it has to be true.
    """

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_the_job_exits_nonzero_when_a_config_fails(self, mock_run_sync, mock_get_config):
        mock_get_config.return_value = [{'latitude': 51.5, 'longitude': -0.1}]
        mock_run_sync.side_effect = Exception("credentials expired")

        with self.assertRaises(SystemExit) as ctx:
            fetch_weekly.main()

        self.assertNotEqual(ctx.exception.code, 0)

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_the_failure_message_says_how_many_areas_failed(self, mock_run_sync, mock_get_config):
        """Three of three failing is an outage; one of three is a bad search
        area. The exit message is the whole of what an operator sees first."""
        mock_get_config.return_value = [
            {'latitude': 51.5, 'longitude': -0.1},
            {'latitude': 52.5, 'longitude': -0.2},
            {'latitude': 53.5, 'longitude': -0.3},
        ]
        mock_run_sync.side_effect = [Exception("boom"), None, Exception("boom")]

        with self.assertRaises(SystemExit) as ctx:
            fetch_weekly.main()

        self.assertIn('2', str(ctx.exception))
        self.assertIn('3', str(ctx.exception))

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_the_job_exits_cleanly_when_every_config_succeeds(self, mock_run_sync, mock_get_config):
        """The half that keeps this from being a job that always fails."""
        mock_get_config.return_value = [{'latitude': 51.5, 'longitude': -0.1}]
        mock_run_sync.return_value = None

        fetch_weekly.main()  # must not raise

    @patch('app.cron.fetch_weekly.get_config_params')
    def test_an_unreadable_config_table_fails_the_job(self, mock_get_config):
        """Reading `config_search_params` is the first BigQuery call the job
        makes, so it is where a dead credential shows up first."""
        mock_get_config.side_effect = Exception("403 insufficient authentication scopes")

        with self.assertRaises(SystemExit) as ctx:
            fetch_weekly.main()

        self.assertNotEqual(ctx.exception.code, 0)

    @patch('app.cron.fetch_weekly.get_config_params')
    @patch('app.cron.fetch_weekly.run_sync_for_config')
    def test_an_empty_config_table_is_not_a_failure(self, mock_run_sync, mock_get_config):
        """Deliberately not an error. Emptying the table is how the cron gets
        paused, and a paused job that pages every week would just get muted."""
        mock_get_config.return_value = []

        fetch_weekly.main()  # must not raise

        mock_run_sync.assert_not_called()

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_a_failed_append_is_not_a_successful_sync(
        self, mock_append, mock_process, mock_load, mock_fetch
    ):
        """`append_to_bigquery` returns False on failure. The caller logged
        "Append failed." and then returned normally, so the new restaurants
        were dropped and the run reported success."""
        mock_fetch.return_value = []
        mock_load.return_value = set()
        mock_process.return_value = ([{'FHRSID': '1'}], "Summary")
        mock_append.return_value = False

        with self.assertRaises(Exception):
            fetch_weekly.run_sync_for_config({
                'latitude': 51.5074, 'longitude': -0.1278, 'max_results': 5000,
                'target_bq_table': 'p.d.t', 'radius': 5
            })

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_a_successful_append_is_still_quiet(
        self, mock_append, mock_process, mock_load, mock_fetch
    ):
        mock_fetch.return_value = []
        mock_load.return_value = set()
        mock_process.return_value = ([{'FHRSID': '1'}], "Summary")
        mock_append.return_value = True

        fetch_weekly.run_sync_for_config({
            'latitude': 51.5074, 'longitude': -0.1278, 'max_results': 5000,
            'target_bq_table': 'p.d.t', 'radius': 5
        })

    @patch('app.cron.fetch_weekly.fetch_data_for_all_coordinates')
    @patch('app.cron.fetch_weekly.load_fhrsids_from_bq')
    @patch('app.cron.fetch_weekly.process_and_update_master_data')
    @patch('app.cron.fetch_weekly.append_to_bigquery')
    def test_nothing_new_to_append_is_not_a_failure(
        self, mock_append, mock_process, mock_load, mock_fetch
    ):
        """Most weeks find nothing, and that is the pipeline working."""
        mock_fetch.return_value = []
        mock_load.return_value = set()
        mock_process.return_value = ([], "Summary")

        fetch_weekly.run_sync_for_config({
            'latitude': 51.5074, 'longitude': -0.1278, 'max_results': 5000,
            'target_bq_table': 'p.d.t', 'radius': 5
        })

        mock_append.assert_not_called()

if __name__ == '__main__':
    unittest.main()
