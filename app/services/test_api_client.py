import unittest
from unittest.mock import patch, MagicMock
from app.services.api_client import fetch_api_data

class TestApiClient(unittest.TestCase):
    @patch('app.services.api_client.requests.get')
    def test_fetch_api_data_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'success'}
        mock_get.return_value = mock_response

        result = fetch_api_data(0.0, 0.0, 10)
        self.assertEqual(result, {'data': 'success'})

    @patch('app.services.api_client.requests.get')
    def test_fetch_api_data_error_status(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = fetch_api_data(0.0, 0.0, 10)
        self.assertIsNone(result)

    @patch('app.services.api_client.requests.get')
    def test_fetch_api_data_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection Error")

        result = fetch_api_data(0.0, 0.0, 10)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
