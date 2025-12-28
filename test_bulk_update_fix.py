import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from bq_utils import bulk_update_reviews

@patch('bq_utils.write_to_bigquery')
@patch('bq_utils.bigquery.Client')
def test_bulk_update_reviews_exact_match(mock_bq_client, mock_write_to_bq):
    """Test that bulk_update_reviews works with exact lowercase column names."""
    mock_write_to_bq.return_value = True
    mock_query_job = MagicMock()
    mock_bq_client.return_value.query.return_value = mock_query_job
    mock_query_job.result.return_value = None
    mock_query_job.errors = None

    df = pd.DataFrame({
        'fhrsid': ['123'],
        'manual_review': ['accepted']
    })
    
    result = bulk_update_reviews('proj', 'dataset', 'table', df)
    assert result is True
    mock_write_to_bq.assert_called_once()

@patch('bq_utils.write_to_bigquery')
@patch('bq_utils.bigquery.Client')
def test_bulk_update_reviews_case_insensitive_match(mock_bq_client, mock_write_to_bq):
    """
    Test that bulk_update_reviews works with case-insensitive column names.
    This is the RED PHASE test - it is expected to fail before the fix.
    """
    mock_write_to_bq.return_value = True
    mock_query_job = MagicMock()
    mock_bq_client.return_value.query.return_value = mock_query_job
    mock_query_job.result.return_value = None
    mock_query_job.errors = None

    # Common CSV variations
    df = pd.DataFrame({
        'FHRSID': ['123'],
        'Manual_Review': ['rejected']
    })
    
    # We expect this to return True after the fix (it will normalize names)
    # BEFORE THE FIX, it returns False because it can't find 'fhrsid' and 'manual_review'
    result = bulk_update_reviews('proj', 'dataset', 'table', df)
    assert result is True
    
    # Also verify that the DataFrame passed to write_to_bigquery has normalized columns
    passed_df = mock_write_to_bq.call_args.kwargs['df']
    assert 'fhrsid' in passed_df.columns
    assert 'manual_review' in passed_df.columns
