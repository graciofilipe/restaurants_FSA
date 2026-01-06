import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import io
from app.core.data_processing import (
    load_master_data,
    process_and_update_master_data,
    load_data_from_csv
)

class TestCoreLogicRefactor(unittest.TestCase):
    
    def test_load_master_data_pure(self):
        # Mock BQ loader
        mock_loader = MagicMock(return_value=[{'FHRSID': '1', 'name': 'Test'}])
        
        # Call function
        data = load_master_data("p", "d", "t", mock_loader)
        
        self.assertEqual(len(data), 1)

    def test_process_and_update_master_data_pure(self):
        master_data = [{'FHRSID': '1'}]
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{'FHRSID': '2'}]}}}
        
        # Expectation: Returns (new_restaurants, message)
        result = process_and_update_master_data(master_data, api_data)
        
        self.assertIsInstance(result, tuple) 
        self.assertEqual(len(result), 2)
        new_data, message = result
        
        self.assertEqual(len(new_data), 1)
        self.assertIn("Identified 1 unique new restaurant", message)

    def test_load_data_from_csv_pure(self):
        csv_content = '"fhrsid","colA"\n"1","abc"'
        simulated_file = io.StringIO(csv_content)
        
        df = load_data_from_csv(simulated_file)
        
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_data_from_csv_pure_error(self):
        csv_content = '"colX","colA"\n"1","abc"' # Missing fhrsid
        simulated_file = io.StringIO(csv_content)
        
        # Expectation: Raises ValueError
        with self.assertRaises(ValueError):
            load_data_from_csv(simulated_file)

if __name__ == '__main__':
    unittest.main()