
import unittest
from unittest.mock import MagicMock, patch
from google.cloud import bigquery
import scripts.update_bq_schema as update_script

class TestUpdateBQSchema(unittest.TestCase):

    @patch('scripts.update_bq_schema.bigquery.Client')
    def test_recreate_table_with_new_schema(self, mock_client_cls):
        # Setup mock
        mock_client = mock_client_cls.return_value
        
        # Run the update function
        update_script.recreate_config_table()
        
        # Verify delete_table was called
        mock_client.delete_table.assert_called_with(
            f"{update_script.PROJECT_ID}.{update_script.DATASET_ID}.{update_script.CONFIG_TABLE_ID}", 
            not_found_ok=True
        )
        
        # Verify create_table was called
        mock_client.create_table.assert_called()
        
        # Get the table object passed to create_table
        created_table = mock_client.create_table.call_args[0][0]
        
        # Check schema fields
        schema_names = {field.name: field.field_type for field in created_table.schema}
        
        self.assertIn('latitude', schema_names)
        self.assertEqual(schema_names['latitude'], 'FLOAT')
        
        self.assertIn('longitude', schema_names)
        self.assertEqual(schema_names['longitude'], 'FLOAT')
        
        self.assertIn('radius', schema_names)
        self.assertEqual(schema_names['radius'], 'INTEGER')
        
        self.assertIn('max_results', schema_names)
        self.assertEqual(schema_names['max_results'], 'INTEGER')
        
        self.assertIn('target_bq_table', schema_names)
        self.assertEqual(schema_names['target_bq_table'], 'STRING')

if __name__ == '__main__':
    unittest.main()
