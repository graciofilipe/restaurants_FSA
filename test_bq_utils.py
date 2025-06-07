import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from bq_utils import read_from_bigquery  # Ensure this import matches your file structure

# Define a fixture for BigQuery client options if needed, or use a default
@pytest.fixture
def mock_bigquery_client():
    with patch('google.cloud.bigquery.Client') as mock_client_constructor:
        mock_client_instance = MagicMock()
        mock_client_constructor.return_value = mock_client_instance

        # Mock the query method
        mock_query_job = MagicMock()
        mock_client_instance.query.return_value = mock_query_job

        # Mock the to_dataframe method to return an empty DataFrame
        # This is the crucial part to test if db-dtypes is found and usable by pandas
        mock_query_job.to_dataframe.return_value = pd.DataFrame({'some_column': [1, 2]})

        yield mock_client_instance

def test_read_from_bigquery_calls_to_dataframe(mock_bigquery_client):
    '''
    Test that read_from_bigquery successfully calls to_dataframe()
    which would have failed if db-dtypes was not present.
    '''
    project_id = "test-project"
    dataset_id = "test-dataset"
    table_id = "test-table"
    fhrsid_list = ["12345"]

    # Call the function that uses to_dataframe()
    # We are not interested in the actual data, but that the call itself doesn't raise an ImportError
    try:
        df = read_from_bigquery(fhrsid_list, project_id, dataset_id, table_id)
        # Check if a DataFrame (even if empty or mock) is returned
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        # Check if to_dataframe() was called on the query job.
        mock_bigquery_client.query.return_value.to_dataframe.assert_called_once()
    except ImportError:
        pytest.fail("ImportError was raised, db-dtypes might still be missing or not importable.")
    except Exception as e:
        pytest.fail(f"read_from_bigquery raised an unexpected exception: {e}")
