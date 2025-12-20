import unittest
from unittest.mock import patch, MagicMock
from st_app import main_ui

class TestAppAuthIntegration(unittest.TestCase):
    @patch('st_app.AuthManager')
    @patch('st_app.login_page')
    @patch('st_app.st')
    def test_main_ui_unauthenticated_redirects(self, mock_st, mock_login_page, mock_auth_manager_cls):
        # Setup unauthenticated state
        mock_auth = mock_auth_manager_cls.return_value
        mock_auth.is_authenticated.return_value = False
        
        main_ui()
        
        # Should call login_page and NOT show the title
        mock_login_page.assert_called_once()
        mock_st.title.assert_not_called()

    @patch('st_app.AuthManager')
    @patch('st_app.login_page')
    @patch('st_app.st')
    def test_main_ui_authenticated_shows_app(self, mock_st, mock_login_page, mock_auth_manager_cls):
        # Setup authenticated state
        mock_auth = mock_auth_manager_cls.return_value
        mock_auth.is_authenticated.return_value = True
        mock_auth.get_user_email.return_value = 'test@example.com'
        
        # Side effect to return list of mocks based on input
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
        
        main_ui()
        
        # Should NOT call login_page and SHOULD show the title
        mock_login_page.assert_not_called()
        mock_st.title.assert_called_with("Food Standards Agency API Explorer")

if __name__ == '__main__':
    unittest.main()
