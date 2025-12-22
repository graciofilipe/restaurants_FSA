import unittest
from unittest.mock import patch, MagicMock
import os

# We'll assume the AuthManager will be in auth/firebase_auth.py
# and we'll mock streamlit and firebase_admin
import sys

# Mocking streamlit before it's imported in the module under test
mock_st = MagicMock()
sys.modules['streamlit'] = mock_st

from auth.firebase_auth import AuthManager

class TestAuthManager(unittest.TestCase):
    def setUp(self):
        # Clear mock calls between tests
        mock_st.reset_mock()
        # Setup basic secrets mock
        mock_st.secrets = {
            "firebase": {
                "projectId": "test-project",
                "apiKey": "test-key"
            }
        }
        # Clear session state mock
        mock_st.session_state = {}

    @patch('auth.firebase_auth.auth.verify_id_token')
    def test_verify_token_success(self, mock_verify):
        """Tests that a valid token correctly authenticates a user."""
        mock_verify.return_value = {
            'email': 'test@user.com',
            'uid': '12345'
        }
        
        manager = AuthManager()
        result = manager.verify_token("valid-token")
        
        self.assertTrue(result)
        self.assertEqual(mock_st.session_state['user_email'], 'test@user.com')
        self.assertTrue(mock_st.session_state['authenticated'])

    @patch('auth.firebase_auth.auth.verify_id_token')
    def test_verify_token_failure(self, mock_verify):
        """Tests that an invalid token does not authenticate a user."""
        mock_verify.side_effect = Exception("Invalid token")
        
        manager = AuthManager()
        result = manager.verify_token("invalid-token")
        
        self.assertFalse(result)
        self.assertNotIn('user_email', mock_st.session_state)
        self.assertFalse(mock_st.session_state.get('authenticated', False))

    def test_is_authenticated_logic(self):
        """Tests the is_authenticated helper."""
        manager = AuthManager()
        
        # Initially False
        self.assertFalse(manager.is_authenticated())
        
        # Set authenticated
        mock_st.session_state['authenticated'] = True
        self.assertTrue(manager.is_authenticated())

    def test_sign_out(self):
        """Tests that sign_out clears the session state."""
        mock_st.session_state['authenticated'] = True
        mock_st.session_state['user_email'] = 'test@user.com'
        
        manager = AuthManager()
        manager.sign_out()
        
        self.assertFalse(mock_st.session_state.get('authenticated', False))
        self.assertNotIn('user_email', mock_st.session_state)

if __name__ == "__main__":
    unittest.main()
