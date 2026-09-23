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

    def test_the_bigquery_path_is_not_user_editable(self):
        """The sidebar offered a "BigQuery Table Path" box whose value was
        never read -- `bq_path` is assigned from the constant two lines above
        it and nothing reassigns it. So the control did nothing, while looking
        like it retargeted the whole app.

        It is also where an injected table path would have entered: the query
        builders interpolate this by f-string. Removing the widget closes that
        route without touching the broader SQL-construction question.
        """
        with open('app/ui/st_app.py', 'r') as f:
            content = f.read()
        assert 'BigQuery Table Path' not in content
        assert 'bq_path_input' not in content


if __name__ == '__main__':
    unittest.main()