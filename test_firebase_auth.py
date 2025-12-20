import unittest
from unittest.mock import patch, MagicMock
from auth.firebase_auth import AuthManager

class TestFirebaseAuth(unittest.TestCase):
    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_is_authenticated_false(self, mock_st, mock_stx):
        # Setup mock session state
        mock_st.session_state = {}
        
        auth = AuthManager()
        self.assertFalse(auth.is_authenticated())

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_is_authenticated_true(self, mock_st, mock_stx):
        # Setup mock session state with a user
        mock_st.session_state = {'user': {'email': 'test@example.com'}}
        
        auth = AuthManager()
        self.assertTrue(auth.is_authenticated())

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_get_user_email(self, mock_st, mock_stx):
        mock_st.session_state = {'user': {'email': 'test@example.com'}}
        
        auth = AuthManager()
        self.assertEqual(auth.get_user_email(), 'test@example.com')

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_get_user_email_none(self, mock_st, mock_stx):
        mock_st.session_state = {}
        
        auth = AuthManager()
        self.assertIsNone(auth.get_user_email())

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_sign_out(self, mock_st, mock_stx):
        mock_st.session_state = {'user': {'email': 'test@example.com'}}
        mock_cookie_manager = mock_stx.return_value
        
        auth = AuthManager()
        auth.sign_out()
        
        self.assertIsNone(mock_st.session_state['user'])
        mock_cookie_manager.delete.assert_called_with('auth_user')

    @patch('auth.firebase_auth.firebase_auth.verify_id_token')
    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_check_auth_query_params(self, mock_st, mock_stx, mock_verify):
        mock_st.session_state = {}
        mock_st.query_params = {'token': 'test-token', 'email': 'test@example.com'}
        mock_cookie_manager = mock_stx.return_value
        
        # Mock verification success
        mock_verify.return_value = {'email': 'test@example.com'}
        
        auth = AuthManager()
        # Mock rerun since it stops execution
        mock_st.rerun.side_effect = Exception("Rerun called")
        
        try:
            auth.check_auth()
        except Exception as e:
            self.assertEqual(str(e), "Rerun called")
        
        self.assertEqual(mock_st.session_state['user'], {'email': 'test@example.com', 'token': 'test-token'})
        mock_cookie_manager.set.assert_called()

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_check_auth_cookie(self, mock_st, mock_stx):
        mock_st.session_state = {}
        mock_st.query_params = {}
        mock_cookie_manager = mock_stx.return_value
        mock_cookie_manager.get.return_value = '{"email": "cookie@example.com", "token": "cookie-token"}'
        
        auth = AuthManager()
        auth.check_auth()
        
        self.assertEqual(mock_st.session_state['user'], {'email': 'cookie@example.com', 'token': 'cookie-token'})

    @patch('auth.firebase_auth.stx.CookieManager')
    @patch('auth.firebase_auth.st')
    def test_login_button(self, mock_st, mock_stx):
        mock_st.secrets = {"firebase": {"apiKey": "key", "authDomain": "domain", "projectId": "id"}}
        
        auth = AuthManager()
        auth.login_button()
        
        mock_st.components.v1.html.assert_called()

if __name__ == '__main__':
    unittest.main()
