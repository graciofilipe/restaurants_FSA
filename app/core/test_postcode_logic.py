import pandas as pd
import pytest
from app.core.data_processing import add_outcode_column

def test_add_outcode_column_basic():
    """Test that the outcode is correctly extracted from valid postcodes."""
    data = {
        'PostCode': ['SE1 7PB', 'SW16 5RT', 'E1 6AN', 'M1 1AA']
    }
    df = pd.DataFrame(data)
    
    df_processed = add_outcode_column(df)
    
    assert 'outcode' in df_processed.columns
    expected_outcodes = ['SE1', 'SW16', 'E1', 'M1']
    assert df_processed['outcode'].tolist() == expected_outcodes

def test_add_outcode_column_edge_cases():
    """Test handling of missing values, no spaces, and empty strings."""
    data = {
        'PostCode': [
            'SE14 5QR',  # Valid
            'INVALID',   # No space
            '',          # Empty string
            None,        # Missing
            'W1A 1AA'    # Valid
        ]
    }
    df = pd.DataFrame(data)
    
    df_processed = add_outcode_column(df)
    
    # Check "INVALID" case - assuming simple split logic as decided
    assert df_processed.iloc[1]['outcode'] == 'INVALID' 
    
    # Check empty string
    assert df_processed.iloc[2]['outcode'] == ''
    
    # Check None - should probably result in None or NaN
    assert pd.isna(df_processed.iloc[3]['outcode'])
    
    # Check SE14 is not confused with SE1 (implied by extraction)
    assert df_processed.iloc[0]['outcode'] == 'SE14'

def test_add_outcode_column_preserves_data():
    """Ensure the original dataframe data is preserved."""
    data = {'PostCode': ['SE1 7PB'], 'RatingValue': ['5']}
    df = pd.DataFrame(data)
    
    df_processed = add_outcode_column(df)
    
    assert 'RatingValue' in df_processed.columns
    assert df_processed['RatingValue'].iloc[0] == '5'

def test_add_outcode_column_empty_df():
    """Test with empty dataframe."""
    df = pd.DataFrame(columns=['PostCode'])
    df_processed = add_outcode_column(df)
    assert 'outcode' in df_processed.columns
    assert len(df_processed) == 0
