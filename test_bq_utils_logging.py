
import unittest
from unittest.mock import MagicMock, patch
import logging
from app.services.bq_utils import get_distinct_local_authorities

class TestBQUtilsLogging(unittest.TestCase):
    @patch('app.services.bq_utils.pandas_gbq.read_gbq')
    @patch('app.services.bq_utils.logger')
    def test_get_distinct_local_authorities_logs_count(self, mock_logger, mock_read_gbq):
        # Setup mock return value
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__.return_value = 2
        mock_df.__getitem__.return_value.tolist.return_value = ['Authority A', 'Authority B']
        mock_read_gbq.return_value = mock_df

        # Call the function
        project_id = 'test-project'
        dataset_id = 'test-dataset'
        table_id = 'test-table'
        get_distinct_local_authorities(project_id, dataset_id, table_id)

        # Assert logging
        expected_log_message = f"Fetched 2 distinct Local Authorities from {project_id}.{dataset_id}.{table_id}."
        mock_logger.info.assert_any_call(expected_log_message)

if __name__ == '__main__':
    unittest.main()
