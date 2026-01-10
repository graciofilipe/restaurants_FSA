import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from app.services.bq_utils import bulk_update_reviews

class TestBulkStatusUpdate(unittest.TestCase):
    
    @patch('app.services.bq_utils.bigquery.Client')
    @patch('app.services.bq_utils.write_to_bigquery')
    def test_bulk_update_reviews_logic(self, mock_write, mock_client_cls):
        # Setup mocks
        mock_client = mock_client_cls.return_value
        mock_query_job = MagicMock()
        mock_client.query.return_value = mock_query_job
        mock_query_job.result.return_value = None
        mock_query_job.num_dml_affected_rows = 5
        
        mock_write.return_value = True

        # Simulate UI input
        selected_fhrsids = ["1", "2", "3"]
        new_status = "accepted"
        project_id = "test-project"
        dataset_id = "test-dataset"
        table_id = "test-table"

        # Logic to be implemented in UI or helper
        # Create DataFrame
        df_update = pd.DataFrame({
            'fhrsid': selected_fhrsids,
            'manual_review': [new_status] * len(selected_fhrsids)
        })

        # Call existing service function
        success, message = bulk_update_reviews(project_id, dataset_id, table_id, df_update)

        # Assertions
        self.assertTrue(success)
        self.assertIn("5 rows updated", message)
        
        # Verify write_to_bigquery called with correct DataFrame
        mock_write.assert_called_once()
        args, kwargs = mock_write.call_args
        df_passed = kwargs.get('df') if 'df' in kwargs else args[0]
        
        self.assertEqual(len(df_passed), 3)
        self.assertEqual(list(df_passed['fhrsid']), selected_fhrsids)
        self.assertEqual(list(df_passed['manual_review']), [new_status] * 3)

    @patch('app.ui.bulk_update.st')
    @patch('app.ui.bulk_update.bulk_update_reviews')
    def test_render_bulk_update_ui_triggers_update(self, mock_bulk_update, mock_st):
        # Import inside test to avoid early import error before file exists
        from app.ui.bulk_update import render_bulk_update_ui
        
        # Setup inputs
        selected_rows = [{'fhrsid': '1'}, {'fhrsid': '2'}]
        project_id = "p"
        dataset_id = "d"
        table_id = "t"
        
        # Setup mocks
        mock_st.selectbox.return_value = "accepted" # User selects "accepted"
        mock_st.button.return_value = True # User clicks "Update"
        mock_bulk_update.return_value = (True, "Updated 2 rows")
        
        # Execute
        render_bulk_update_ui(project_id, dataset_id, table_id, selected_rows)
        
        # Verify
        mock_st.selectbox.assert_called()
        mock_st.button.assert_called()
        mock_bulk_update.assert_called_once()
        
        # Verify DataFrame passed to bulk_update
        args, kwargs = mock_bulk_update.call_args
        df_passed = args[3]
        self.assertEqual(len(df_passed), 2)
        self.assertEqual(list(df_passed['manual_review']), ["accepted", "accepted"])
        
        # Verify success message
        mock_st.success.assert_called_with("Updated 2 rows")
        mock_st.rerun.assert_called_once()

if __name__ == '__main__':
    unittest.main()
