import unittest
from unittest.mock import patch, MagicMock
from auth.firebase_auth import AuthManager
import firebase_admin

class TestAuthVerification(unittest.TestCase):
    
    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    @patch('auth.firebase_auth.firebase_admin.initialize_app')
    @patch('auth.firebase_auth.firebase_auth.verify_id_token')
    def test_verify_token_called_and_project_id_configured(self, mock_verify, mock_init, mock_st, mock_stx):
        # Setup secrets
        mock_st.secrets = {
            "firebase": {
                "projectId": "podcast-ce10c",
                "apiKey": "fake-key",
                "authDomain": "fake.firebaseapp.com"
            }
        }
        
        # Setup session state and params
        mock_st.session_state = {}
        mock_st.query_params = {'token': 'valid-token', 'email': 'test@example.com'}
        
        # Mock cookie manager
        mock_cookie_manager = mock_stx.return_value
        
        # Mock verify to return decoded token
        mock_verify.return_value = {'email': 'test@example.com', 'uid': '123'}
        
        # Initialize AuthManager
        # We need to force re-initialization logic to test it, 
        # but AuthManager.__init__ checks firebase_admin._apps.
        # We'll mock _apps to be empty initially.
        with patch('auth.firebase_auth.firebase_admin._apps', []):
            auth = AuthManager()
            
            # Check if initialize_app was called with correct project ID
            # Current code calls it without args, so this assertion should FAIL
            mock_init.assert_called_with(options={'projectId': 'podcast-ce10c'})

        # Mock rerun to avoid error
        mock_st.rerun.side_effect = Exception("Rerun")
        
        try:
            auth.check_auth()
        except Exception:
            pass
            
        # Check if verify_id_token was called
        # Current code doesn't call it, so this should FAIL
        mock_verify.assert_called_with('valid-token')
        
        # Check if user is set (it currently IS set, but strictly because of the other failures, we want to ensure it's conditional)
        self.assertEqual(mock_st.session_state['user']['email'], 'test@example.com')

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    @patch('auth.firebase_auth.firebase_auth.verify_id_token')
    def test_check_auth_fails_invalid_token(self, mock_verify, mock_st, mock_stx):
        # Setup secrets
        mock_st.secrets = {"firebase": {"projectId": "podcast-ce10c"}}
        
        # Setup params
        mock_st.session_state = {}
        mock_st.query_params = {'token': 'invalid-token', 'email': 'test@example.com'}
        
        # Mock verify to raise error
        mock_verify.side_effect = ValueError("Invalid token")
        
        auth = AuthManager()
        
        # Run check_auth
        auth.check_auth()
        
        # Assert user is NOT set
        # Current code sets it blindly, so this should FAIL
        self.assertIsNone(mock_st.session_state.get('user'))

if __name__ == '__main__':
    unittest.main()
