import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
from st_app import display_new_restaurants

class TestUIIntegration(unittest.TestCase):
    @patch('st_app.st')
    def test_display_new_restaurants_link_column(self, mock_st):
        # Setup data
        new_restaurants = [
            {'FHRSID': '1', 'BusinessName': 'Test', 'Maps Link': 'http://maps.google.com'}
        ]
        
        display_new_restaurants(new_restaurants)
        
        # Verify dataframe call
        # We expect st.dataframe to be called with a DataFrame and column_config
        args, kwargs = mock_st.dataframe.call_args
        
        df_arg = args[0]
        self.assertIsInstance(df_arg, pd.DataFrame)
        self.assertIn('Maps Link', df_arg.columns)
        
        # Check column_config
        column_config = kwargs.get('column_config')
        self.assertIsNotNone(column_config)
        self.assertIn('Maps Link', column_config)
        
        # Check that it's a LinkColumn (we can check the type name or properties if mocked)
        # Since we mock st, st.column_config.LinkColumn is a mock.
        # We can check if it was called.
        mock_st.column_config.LinkColumn.assert_called_with(
            "Research on Maps", display_text="Search Maps"
        )

if __name__ == '__main__':
    unittest.main()