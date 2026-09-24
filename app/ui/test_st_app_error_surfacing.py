"""The UI has to tell "nothing matched" apart from "the read failed".

These two outcomes took the same path through `load_data_into_state` until D9,
because `load_filtered_data_from_bq` caught its own exceptions and returned an
empty list. The `except` clause in `load_data_into_state` and the two around
the sidebar dropdowns were unreachable code -- present, correct, and never
executed. A 403 rendered as "No data found matching criteria."

That message is worse than unhelpful: it tells the user the filters are too
narrow, so the reasonable next action is to widen them, and no amount of
widening fixes an expired token.
"""
import unittest
from unittest.mock import MagicMock, patch

from app.services.bq_utils import BigQueryExecutionError


class TestTheUiTellsTheTwoApart(unittest.TestCase):

    def _load(self, loader_result=None, loader_error=None):
        """Run `load_data_into_state` against a mocked Streamlit and loader."""
        from app.ui import st_app

        with patch.object(st_app, 'st') as mock_st, \
             patch.object(st_app, 'load_filtered_data_from_bq') as mock_loader:
            mock_st.session_state = MagicMock()
            if loader_error is not None:
                mock_loader.side_effect = loader_error
            else:
                mock_loader.return_value = loader_result or []

            st_app.load_data_into_state('p', 'd', 't', None, None)

        return mock_st

    def test_a_failed_read_is_reported_as_an_error(self):
        st = self._load(loader_error=BigQueryExecutionError(
            "Could not load data from proj.ds.tbl: 403 insufficient authentication scopes"))

        st.error.assert_called_once()
        message = st.error.call_args.args[0]
        self.assertIn('403', message)
        self.assertIn('proj.ds.tbl', message)

    def test_a_failed_read_is_not_reported_as_an_empty_one(self):
        """The regression this is really guarding. Before D9 this assertion
        failed: the user got a warning about their filters."""
        st = self._load(loader_error=BigQueryExecutionError("403"))

        st.warning.assert_not_called()

    def test_a_genuinely_empty_result_still_warns_about_the_filters(self):
        """The converse. Most empty loads really are over-narrow filters, and
        that advice is correct -- it just has to stop being the only answer."""
        st = self._load(loader_result=[])

        st.warning.assert_called_once()
        self.assertIn('No data found', st.warning.call_args.args[0])
        st.error.assert_not_called()

    def test_a_failed_read_does_not_mark_the_data_as_loaded(self):
        """`data_loaded` gates the rest of the page. Setting it after a failed
        read would leave the previous frame on screen under fresh filters."""
        from types import SimpleNamespace

        from app.ui import st_app

        state = SimpleNamespace()
        with patch.object(st_app, 'st') as mock_st, \
             patch.object(st_app, 'load_filtered_data_from_bq') as mock_loader:
            mock_st.session_state = state
            mock_loader.side_effect = BigQueryExecutionError("403")

            st_app.load_data_into_state('p', 'd', 't', None, None)

        self.assertFalse(hasattr(state, 'data_loaded'))


class TestTheSidebarDropdownsSurviveAFailure(unittest.TestCase):
    """The dropdown helpers raise too, but the sidebar renders before the user
    can press anything, so an exception there would take the whole page down
    and leave nowhere to read the message. Those two call sites catch, report
    and fall back to an empty list -- which is honest, because an empty
    dropdown next to a visible error is not claiming the table is empty.
    """

    def test_both_dropdown_call_sites_are_guarded(self):
        with open('app/ui/st_app.py', 'r') as f:
            content = f.read()
        self.assertIn('Failed to fetch outcodes', content)
        self.assertIn('Failed to fetch local authorities', content)


if __name__ == '__main__':
    unittest.main()
