import unittest

class TestStAppAccessibility(unittest.TestCase):
    def test_st_app_no_auth_imports(self):
        """Verify st_app.py no longer imports AuthManager or login_page"""
        with open('app/ui/st_app.py', 'r') as f:
            content = f.read()
        assert 'from auth.firebase_auth import AuthManager' not in content
        assert 'from login import login_page' not in content

    def test_st_app_main_ui_no_auth_check(self):
        """Verify main_ui in st_app.py does not call auth_manager.is_authenticated()"""
        with open('app/ui/st_app.py', 'r') as f:
            content = f.read()
        assert 'auth_manager.is_authenticated()' not in content
        assert 'login_page(auth_manager)' not in content

if __name__ == '__main__':
    unittest.main()