import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
from google.cloud import bigquery
import recent_restaurant_analysis # Import the module to be tested

# Helper function to compare lists of SchemaField objects
def assert_schema_fields_equal(schema1, schema2):
    if len(schema1) != len(schema2):
        raise AssertionError(f"Schema lengths differ: {len(schema1)} != {len(schema2)}")
    for field1, field2 in zip(schema1, schema2):
        if not (field1.name == field2.name and field1.field_type == field2.field_type):
            raise AssertionError(f"SchemaField mismatch: {field1} != {field2}")
    return True

class TestRecentRestaurantAnalysis(unittest.TestCase):

    @patch('recent_restaurant_analysis.st')
    @patch('recent_restaurant_analysis.bq_utils.write_to_bigquery')
    # No longer need to mock get_recent_restaurants here as we pass the df directly
    def test_create_recent_restaurants_temp_table_flow(self, mock_write_to_bigquery, mock_st):
        # 1. Define mock project_id and dataset_id (as st.secrets is not directly used by the new function)
        test_project_id = "test_project"
        test_dataset_id = "test_dataset"

        # 2. Create a sample DataFrame
        sample_data = {
            'FHRSID': [1, 2, 3],
            'BusinessName': ['Cafe One', 'Restaurant Two', 'Bakery Three'],
            'RatingValue': ['5', '4', '5'],
            'InspectionDate': pd.to_datetime(['2023-01-01', '2023-02-15', '2023-03-10']),
            'Score': [10.5, 8.2, 9.9],
            'IsNew': [True, False, True]
        }
        sample_df = pd.DataFrame(sample_data)

        # Expected schema based on sample_df
        expected_bq_schema = [
            bigquery.SchemaField(name='FHRSID', field_type='INTEGER'),
            bigquery.SchemaField(name='BusinessName', field_type='STRING'),
            bigquery.SchemaField(name='RatingValue', field_type='STRING'),
            bigquery.SchemaField(name='InspectionDate', field_type='TIMESTAMP'),
            bigquery.SchemaField(name='Score', field_type='FLOAT'),
            bigquery.SchemaField(name='IsNew', field_type='BOOLEAN')
        ]

        # 3. Call the refactored function
        mock_write_to_bigquery.return_value = True # Simulate successful write
        recent_restaurant_analysis.create_recent_restaurants_temp_table(
            restaurants_df=sample_df,
            project_id=test_project_id,
            dataset_id=test_dataset_id
        )

        # 4. Assert that bq_utils.write_to_bigquery was called correctly
        args, kwargs = mock_write_to_bigquery.call_args

        pd.testing.assert_frame_equal(kwargs['df'], sample_df)
        self.assertEqual(kwargs['project_id'], test_project_id)
        self.assertEqual(kwargs['dataset_id'], test_dataset_id)
        self.assertEqual(kwargs['table_id'], "recent_restaurants_temp")
        self.assertEqual(kwargs['columns_to_select'], sample_df.columns.tolist())

        # Compare schemas
        assert_schema_fields_equal(kwargs['bq_schema'], expected_bq_schema)

        # 5. Assert st.success was called (since mock_write_to_bigquery.return_value = True)
        mock_st.success.assert_called_with(f"Successfully wrote data to BigQuery temporary table: {test_project_id}.{test_dataset_id}.recent_restaurants_temp")
        mock_st.error.assert_not_called()

    @patch('recent_restaurant_analysis.st')
    @patch('recent_restaurant_analysis.bq_utils.write_to_bigquery')
    def test_create_recent_restaurants_temp_table_handles_empty_df(self, mock_write_to_bigquery, mock_st):
        test_project_id = "test_project_empty"
        test_dataset_id = "test_dataset_empty"
        empty_df = pd.DataFrame()

        recent_restaurant_analysis.create_recent_restaurants_temp_table(
            restaurants_df=empty_df,
            project_id=test_project_id,
            dataset_id=test_dataset_id
        )

        mock_st.warning.assert_called_with("No restaurant data provided to create_recent_restaurants_temp_table. Skipping table creation.")
        mock_write_to_bigquery.assert_not_called()
        mock_st.success.assert_not_called()
        mock_st.error.assert_not_called()

    @patch('recent_restaurant_analysis.st')
    @patch('recent_restaurant_analysis.bq_utils.write_to_bigquery')
    def test_create_recent_restaurants_temp_table_handles_bq_write_failure(self, mock_write_to_bigquery, mock_st):
        test_project_id = "test_project_fail"
        test_dataset_id = "test_dataset_fail"
        sample_df = pd.DataFrame({'col1': [1]}) # Minimal non-empty DF

        mock_write_to_bigquery.return_value = False # Simulate failed write

        recent_restaurant_analysis.create_recent_restaurants_temp_table(
            restaurants_df=sample_df,
            project_id=test_project_id,
            dataset_id=test_dataset_id
        )

        mock_write_to_bigquery.assert_called_once() # Ensure it was called
        mock_st.error.assert_called_with(f"Failed to write data to BigQuery temporary table: {test_project_id}.{test_dataset_id}.recent_restaurants_temp")
        mock_st.success.assert_not_called()

    def test_pandas_dtype_to_bq_type_mapping(self):
        # Test the helper function directly
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.Series([1, 2]).dtype), 'INTEGER')
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.Series([1.0, 2.0]).dtype), 'FLOAT')
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.Series([True, False]).dtype), 'BOOLEAN')
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.Series(['a', 'b']).dtype), 'STRING')
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.to_datetime(pd.Series(['2023-01-01']))).dtype, 'TIMESTAMP')
        # Test for a generic object type that should default to STRING
        self.assertEqual(recent_restaurant_analysis.pandas_dtype_to_bq_type(pd.Series([{'a': 1}, {'b': 2}]).dtype), 'STRING')

if __name__ == '__main__':
    unittest.main()
