import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import pandas as pd
import streamlit as st # We will mock this heavily

# Import functions from st_app.py
# Assuming st_app.py is in the same directory or accessible via PYTHONPATH
from st_app import fhrsid_lookup_logic, main_ui
# We also need to patch functions imported by st_app, like read_from_bigquery and update_manual_review

# Helper class to simulate streamlit.SessionState more directly
class MockSessionState:
    def __init__(self, state_dict):
        self._state = state_dict

    def __getitem__(self, key):
        if key not in self._state:
            if key == 'fhrsid_df': return None
            if key == 'successful_fhrsids': return []
            raise KeyError(key)
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __contains__(self, key):
        return key in self._state

    def __getattr__(self, key):
        if key == '_state':
            return super().__getattribute__(key)

        if key not in self._state:
            if key == 'fhrsid_df': return None
            if key == 'successful_fhrsids': return []
            return None

        return self._state.get(key)

    def __setattr__(self, key, value):
        if key == '_state':
            object.__setattr__(self, key, value)
            return
        self._state[key] = value

class TestFhrsidLookupAndUpdateWorkflow(unittest.TestCase):

    def setUp(self):
        # Mock st.session_state for each test
        # We'll patch 'st_app.st' which is the alias used in st_app.py
        # The mock_st_object will then have a 'session_state' attribute we can control.
        
        # Basic session state structure that will be fresh for each test via _run_test_with_patches
        self.base_session_state_dict = {
            'fhrsid_df': None,
            'successful_fhrsids': [],
            'fhrsid_input_str_ui': "",
            'bq_table_lookup_input_str_ui': ""
        }
        # Each test starts with self.current_mock_session_state_dict as a fresh copy of base.
        self.current_mock_session_state_dict = self.base_session_state_dict.copy()


    # Helper to apply common patches and manage session_state
    def _run_test_with_patches(self, test_logic_func, mock_st_extras=None):
        # self.current_mock_session_state_dict is now prepared by setUp and can be
        # further modified by the test method before calling this helper.

        with patch('st_app.st', autospec=True) as mock_st_global, \
             patch('st_app.read_from_bigquery') as mock_read_from_bq, \
             patch('st_app.update_manual_review') as mock_update_review, \
             patch('st_app.pd.concat') as mock_pd_concat:

            # Instantiate our custom MockSessionState
            # It directly uses and modifies self.current_mock_session_state_dict
            mock_st_global.session_state = MockSessionState(self.current_mock_session_state_dict)

            # The MockSessionState class now handles the getitem, setitem, getattr, setattr logic.
            # No need to set side_effects for these on mock_st_global.session_state anymore.

            # Initialize attributes on MockSessionState instance for any values already in current_mock_session_state_dict.
            # This ensures that if tests set initial values in self.current_mock_session_state_dict
            # before _run_test_with_patches, they are reflected in the MockSessionState object.
            # Note: The MockSessionState constructor already links it to self.current_mock_session_state_dict,
            # so direct manipulation of self.current_mock_session_state_dict (like in some test setups)
            # will be reflected. This loop might be redundant if all setup is done on current_mock_session_state_dict
            # and not by trying to setattr on the session_state object before it's fully mocked.
            # However, it's safer to ensure consistency.
            for k, v in self.current_mock_session_state_dict.items():
                setattr(mock_st_global.session_state, k, v)


            if mock_st_extras:
                mock_st_extras(mock_st_global)

            test_logic_func(mock_st_global, mock_read_from_bq, mock_update_review, mock_pd_concat)


    # --- Tests for fhrsid_lookup_logic ---
    def test_fhrsid_lookup_populates_session_state_on_success(self):
        sample_fhrsid = "12345"
        sample_df = pd.DataFrame({'fhrsid': [sample_fhrsid], 'data': ['test']})
        
        def logic(mock_st, mock_read_from_bq, _, __):
            mock_read_from_bq.return_value = sample_df

            fhrsid_lookup_logic(sample_fhrsid, "proj.dset.tbl", mock_st, mock_read_from_bq, None)

            self.assertIsNotNone(self.current_mock_session_state_dict['fhrsid_df'])
            self.assertEqual(self.current_mock_session_state_dict['fhrsid_df']['fhrsid'].iloc[0], sample_fhrsid)
            self.assertEqual(self.current_mock_session_state_dict['successful_fhrsids'], [sample_fhrsid])
            mock_st.success.assert_called_with(f"Data found for FHRSIDs: {sample_fhrsid}")

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_clears_session_state_on_failure(self):
        def logic(mock_st, mock_read_from_bq, _, __):
            mock_read_from_bq.return_value = None

            self.current_mock_session_state_dict['fhrsid_df'] = pd.DataFrame({'fhrsid': ["old"], 'data': ['old_data']})
            self.current_mock_session_state_dict['successful_fhrsids'] = ["old"]
            # Ensure these are on the mock_st.session_state object too if getattr/setattr used by SUT
            setattr(mock_st.session_state, 'fhrsid_df', self.current_mock_session_state_dict['fhrsid_df'])
            setattr(mock_st.session_state, 'successful_fhrsids', self.current_mock_session_state_dict['successful_fhrsids'])


            fhrsid_lookup_logic("123", "proj.dset.tbl", mock_st, mock_read_from_bq, None)

            self.assertIsNone(self.current_mock_session_state_dict['fhrsid_df'])
            self.assertEqual(self.current_mock_session_state_dict['successful_fhrsids'], [])
            mock_st.warning.assert_called_with("No data found for the provided FHRSIDs: 123 in proj.dset.tbl, or an error occurred during lookup for all specified IDs.")
        
        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_handles_bq_error(self):
        def logic(mock_st, mock_read_from_bq, _, __):
            mock_read_from_bq.side_effect = Exception("BigQuery exploded")

            fhrsid_lookup_logic("123", "proj.dset.tbl", mock_st, mock_read_from_bq, None)

            self.assertIsNone(self.current_mock_session_state_dict['fhrsid_df'])
            self.assertEqual(self.current_mock_session_state_dict['successful_fhrsids'], [])
            mock_st.warning.assert_called_with("No data found for the provided FHRSIDs: 123 in proj.dset.tbl, or an error occurred during lookup for all specified IDs.")

        self._run_test_with_patches(logic)

    def test_main_ui_update_workflow_success_single_fhrsid(self):
        fhrsid = "999"
        bq_path = "proj.dset.tbl"
        new_review_value = "Looks good"
        
        self.current_mock_session_state_dict = {
            'fhrsid_df': pd.DataFrame({'fhrsid': [fhrsid], 'manual_review': ['old_value']}),
            'successful_fhrsids': [fhrsid],
            'fhrsid_input_str_ui': fhrsid,
            'bq_table_lookup_input_str_ui': bq_path
        }

        def mock_st_config(mock_st):
            mock_st.radio.return_value = "FHRSID Lookup"
            button_return_values = {"Update Manual Review": True, "Lookup FHRSIDs": False}
            # Revert to simpler lambda, or keep the def but remove prints.
            # For simplicity, using the original lambda:
            mock_st.button.side_effect = lambda text, key=None: button_return_values.get(text, False)

            # text_input needs to return the bq_path and fhrsid_input when asked for those by main_ui,
            # and the new_review_value for the manual review input.
            def text_input_side_effect(label, value=None, key=None):
                if "New Manual Review Value" in label: return new_review_value
                if "Enter FHRSIDs" in label: return self.current_mock_session_state_dict['fhrsid_input_str_ui']
                if "Enter BigQuery Table Path" in label: return self.current_mock_session_state_dict['bq_table_lookup_input_str_ui']
                return value if value is not None else ""
            mock_st.text_input.side_effect = text_input_side_effect
            mock_st.multiselect.return_value = self.current_mock_session_state_dict['successful_fhrsids']
            # If only one FHRSID, selectbox is not called for FHRSID selection.
            # mock_st.selectbox implicitly won't be called or its call won't matter.

        def logic(mock_st, mock_read_from_bq, mock_update_review, mock_pd_concat):
            mock_update_review.return_value = True
            refreshed_df = pd.DataFrame({'fhrsid': [fhrsid], 'manual_review': [new_review_value]})
            # This mock_read_from_bq will be used by the fhrsid_lookup_logic call during refresh
            mock_read_from_bq.return_value = refreshed_df

            main_ui()

            mock_update_review.assert_called_once_with(
                fhrsid_list=[fhrsid],
                manual_review_value=new_review_value,
                project_id="proj",
                dataset_id="dset",
                table_id="tbl"
            )
            # fhrsid_lookup_logic (which calls read_from_bigquery) is called for refresh
            mock_read_from_bq.assert_called_with([fhrsid], "proj", "dset", "tbl")
            mock_st.success.assert_any_call(f"Manual review updated for FHRSIDs: {fhrsid}. Refreshing data...")
            mock_st.rerun.assert_called_once()
            self.assertEqual(self.current_mock_session_state_dict['fhrsid_df']['manual_review'].iloc[0], new_review_value)

        self._run_test_with_patches(logic, mock_st_config)

    def test_main_ui_update_workflow_update_fails(self):
        fhrsid = "1001"
        bq_path = "proj.dset.tbl"
        new_review_value = "This will fail"

        self.current_mock_session_state_dict = {
            'fhrsid_df': pd.DataFrame({'fhrsid': [fhrsid], 'manual_review': ['initial_state']}),
            'successful_fhrsids': [fhrsid],
            'fhrsid_input_str_ui': fhrsid,
            'bq_table_lookup_input_str_ui': bq_path
        }

        def mock_st_config(mock_st):
            mock_st.radio.return_value = "FHRSID Lookup"
            button_return_values = {"Update Manual Review": True, "Lookup FHRSIDs": False}
            # Revert to simpler lambda
            mock_st.button.side_effect = lambda text, key=None: button_return_values.get(text, False)

            def text_input_side_effect(label, value=None, key=None):
                if "New Manual Review Value" in label: return new_review_value
                if "Enter FHRSIDs" in label: return self.current_mock_session_state_dict['fhrsid_input_str_ui']
                if "Enter BigQuery Table Path" in label: return self.current_mock_session_state_dict['bq_table_lookup_input_str_ui']
                return value if value is not None else ""
            mock_st.text_input.side_effect = text_input_side_effect
            mock_st.multiselect.return_value = self.current_mock_session_state_dict['successful_fhrsids']

        def logic(mock_st, mock_read_from_bq, mock_update_review, mock_pd_concat):
            mock_update_review.return_value = False
            initial_read_call_count = mock_read_from_bq.call_count

            main_ui()

            mock_update_review.assert_called_once_with(
                fhrsid_list=[fhrsid],
                manual_review_value=new_review_value,
                project_id="proj",
                dataset_id="dset",
                table_id="tbl"
            )
            self.assertEqual(mock_read_from_bq.call_count, initial_read_call_count) # No refresh call
            mock_st.rerun.assert_not_called()
            # st.error is called by update_manual_review, so it's implicitly tested via that function's tests.

        self._run_test_with_patches(logic, mock_st_config)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
