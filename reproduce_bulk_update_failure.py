import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from bq_utils import bulk_update_reviews
from google.cloud import bigquery

class TestBulkUpdateReproduction(unittest.TestCase):

    @patch('bq_utils.write_to_bigquery')
    @patch('bq_utils.bigquery.Client')
    def test_bulk_update_reproduction_int_fhrsid(self, mock_bq_client_constructor, mock_write_to_bq):
        # Setup mocks
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_query_job = MagicMock()
        mock_bq_client_instance.query.return_value = mock_query_job
        mock_query_job.result.return_value = None
        mock_query_job.errors = None
        mock_write_to_bq.return_value = True

        df_updates = pd.DataFrame({
            'fhrsid': [123, 456],
            'manual_review': ['accepted', 'rejected']
        })
        result = bulk_update_reviews('p', 'd', 't', df_updates)
        self.assertTrue(result)

    @patch('bq_utils.write_to_bigquery')
    @patch('bq_utils.bigquery.Client')
    def test_bulk_update_reproduction_mismatched_columns(self, mock_bq_client_constructor, mock_write_to_bq):
        # Setup mocks
        mock_bq_client_instance = mock_bq_client_constructor.return_value
        mock_write_to_bq.return_value = True

        # CSV with uppercase columns - SCRIPT EXPECTS lowercase 'fhrsid' and 'manual_review'
        df_updates = pd.DataFrame({
            'FHRSID': [123, 456],
            'Manual_Review': ['accepted', 'rejected']
        })

        project_id = 'test-project'
        dataset_id = 'test-dataset'
        table_id = 'test-table'

        # Execute
        result = bulk_update_reviews(project_id, dataset_id, table_id, df_updates)

        # THIS SHOULD FAIL if it's the root cause
        self.assertFalse(result, "bulk_update_reviews should return False when columns are mismatched")

if __name__ == '__main__':
    unittest.main()