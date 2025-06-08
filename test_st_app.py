import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import pandas as pd
import streamlit as st # We will mock this heavily

# Import functions from st_app.py
# Assuming st_app.py is in the same directory or accessible via PYTHONPATH
from st_app import fhrsid_lookup_logic, main_ui
# We also need to patch functions imported by st_app, like read_from_bigquery and update_manual_review
# Also, BigQueryExecutionError may be raised by read_from_bigquery
from bq_utils import BigQueryExecutionError

# Helper class to simulate streamlit.SessionState more directly
class MockSessionState(dict): # Inherit from dict for easier state management
    def __init__(self, initial_state=None):
        if initial_state is None:
            initial_state = {}
        super().__init__(initial_state)
        # Ensure essential keys are present, mimicking Streamlit's behavior for uninitialized state
        self.setdefault('fhrsid_df', pd.DataFrame()) # Initialize with empty DataFrame
        self.setdefault('successful_fhrsids', [])
        self.setdefault('fhrsid_input_str_ui',"")
        self.setdefault('bq_table_lookup_input_str_ui',"")


    def __getattr__(self, key):
        # Allow attribute-style access for keys that exist
        if key in self:
            return self[key]
        # Default behavior for common Streamlit session_state attributes if not set
        if key == 'fhrsid_df': return pd.DataFrame()
        if key == 'successful_fhrsids': return []
        # Fallback to raising AttributeError if key genuinely doesn't exist and has no default
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}' and no default defined")

    def __setattr__(self, key, value):
        self[key] = value


class TestFhrsidLookupAndUpdateWorkflow(unittest.TestCase):

    def setUp(self):
        # Basic session state structure that will be fresh for each test
        self.base_session_state_dict = {
            'fhrsid_df': pd.DataFrame(), # Default to empty DataFrame
            'successful_fhrsids': [],
            'fhrsid_input_str_ui': "",
            'bq_table_lookup_input_str_ui': ""
        }
        # Each test starts with self.current_mock_session_state as a fresh instance.
        self.current_mock_session_state = MockSessionState(self.base_session_state_dict.copy())

    # Helper to apply common patches and manage session_state
    def _run_test_with_patches(self, test_logic_func, mock_st_extras=None):
        with patch('st_app.st', autospec=True) as mock_st_global, \
             patch('st_app.read_from_bigquery') as mock_read_from_bq, \
             patch('st_app.update_manual_review') as mock_update_review:
             # mock_pd_concat is removed as it's no longer used by fhrsid_lookup_logic

            # Assign our managed MockSessionState instance to the mock_st_global
            mock_st_global.session_state = self.current_mock_session_state

            if mock_st_extras:
                mock_st_extras(mock_st_global)

            # Pass only the relevant mocks to the test logic function
            test_logic_func(mock_st_global, mock_read_from_bq, mock_update_review)


    # --- Tests for fhrsid_lookup_logic ---
    def test_fhrsid_lookup_populates_session_state_on_success(self):
        sample_fhrsid = "12345"
        sample_df = pd.DataFrame({'fhrsid': [sample_fhrsid], 'data': ['test']})
        
        # This test will directly manipulate self.current_mock_session_state
        # which is used by mock_st_global.session_state

        def logic(mock_st, mock_read_from_bq, _): # mock_pd_concat removed
            mock_read_from_bq.return_value = sample_df

            # Call fhrsid_lookup_logic without pd_concat
            fhrsid_lookup_logic(sample_fhrsid, "proj.dset.tbl", mock_st, mock_read_from_bq)

            # Verify read_from_bigquery was called correctly (with a list of strings)
            mock_read_from_bq.assert_called_once_with([sample_fhrsid], "proj", "dset", "tbl")

            self.assertTrue(not self.current_mock_session_state['fhrsid_df'].empty)
            self.assertEqual(self.current_mock_session_state['fhrsid_df']['fhrsid'].iloc[0], sample_fhrsid)
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], [sample_fhrsid])
            mock_st.success.assert_called_with(f"Data found for FHRSIDs: {sample_fhrsid}")

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_multiple_valid_fhrsids(self):
        """Test with multiple valid FHRSIDs, comma-separated."""
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_input = "123,456,789"
            expected_fhrsid_list = ["123", "456", "789"]
            mock_read_from_bq.return_value = pd.DataFrame({'fhrsid': expected_fhrsid_list}) # Simulate finding all

            fhrsid_lookup_logic(fhrsid_input, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_read_from_bq.assert_called_once_with(expected_fhrsid_list, "proj", "dset", "tbl")
            mock_st.success.assert_called_with(f"Data found for FHRSIDs: {', '.join(expected_fhrsid_list)}")
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], expected_fhrsid_list)

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_fhrsids_with_spaces(self):
        """Test FHRSIDs with leading/trailing spaces."""
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_input = " 123 , 456 "
            expected_fhrsid_list = ["123", "456"]
            mock_read_from_bq.return_value = pd.DataFrame({'fhrsid': expected_fhrsid_list})

            fhrsid_lookup_logic(fhrsid_input, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_read_from_bq.assert_called_once_with(expected_fhrsid_list, "proj", "dset", "tbl")
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], expected_fhrsid_list)

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_empty_fhrsid_parts(self):
        """Test with empty parts in FHRSID string due to extra commas."""
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_input = "123,,456," # Trailing comma and empty part
            expected_fhrsid_list = ["123", "456"]
            mock_read_from_bq.return_value = pd.DataFrame({'fhrsid': expected_fhrsid_list})

            fhrsid_lookup_logic(fhrsid_input, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_read_from_bq.assert_called_once_with(expected_fhrsid_list, "proj", "dset", "tbl")
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], expected_fhrsid_list)

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_non_numeric_fhrsid(self):
        """Test with a non-numeric FHRSID in the list."""
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_input = "123,abc,456"

            fhrsid_lookup_logic(fhrsid_input, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_st.error.assert_called_with("Invalid FHRSID: 'abc' is not a valid number. Please enter numeric FHRSIDs only.")
            mock_read_from_bq.assert_not_called()
            self.assertTrue(self.current_mock_session_state['fhrsid_df'].empty)
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], [])

        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_no_data_found(self):
        """ Test fhrsid_lookup_logic when read_from_bigquery returns an empty DataFrame. """
        def logic(mock_st, mock_read_from_bq, _):
            mock_read_from_bq.return_value = pd.DataFrame() # Empty DataFrame

            numeric_fhrsid_for_test = "00000" # Or any other valid numeric string
            fhrsid_lookup_logic(numeric_fhrsid_for_test, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_read_from_bq.assert_called_once_with([numeric_fhrsid_for_test], "proj", "dset", "tbl")
            self.assertTrue(self.current_mock_session_state['fhrsid_df'].empty)
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], [])
            mock_st.warning.assert_called_with(f"No data found for any of the provided FHRSIDs: {numeric_fhrsid_for_test}.")

        self._run_test_with_patches(logic)


    def test_fhrsid_lookup_handles_bq_error(self):
        """ Test fhrsid_lookup_logic when read_from_bigquery raises BigQueryExecutionError. """
        def logic(mock_st, mock_read_from_bq, _):
            error_message = "BigQuery exploded during read"
            mock_read_from_bq.side_effect = BigQueryExecutionError(error_message)
            fhrsid_input = "123" # Keep it simple for error case

            fhrsid_lookup_logic(fhrsid_input, "proj.dset.tbl", mock_st, mock_read_from_bq)

            mock_read_from_bq.assert_called_once_with([fhrsid_input], "proj", "dset", "tbl")
            self.assertTrue(self.current_mock_session_state['fhrsid_df'].empty)
            self.assertEqual(self.current_mock_session_state['successful_fhrsids'], [])
            # Check that st.error was called with the specific error message from BigQueryExecutionError
            mock_st.error.assert_called_with(f"BigQuery error during lookup for FHRSIDs {fhrsid_input}: {error_message}")
        
        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_input_validation_empty_fhrsid(self):
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_lookup_logic("", "proj.dset.tbl", mock_st, mock_read_from_bq)
            mock_st.error.assert_called_with("Please enter one or more FHRSIDs.")
            mock_read_from_bq.assert_not_called()
        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_input_validation_empty_bq_path(self):
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_lookup_logic("123", "", mock_st, mock_read_from_bq)
            mock_st.error.assert_called_with("BigQuery Table Path is required.")
            mock_read_from_bq.assert_not_called()
        self._run_test_with_patches(logic)

    def test_fhrsid_lookup_input_validation_invalid_bq_path_format(self):
        def logic(mock_st, mock_read_from_bq, _):
            fhrsid_lookup_logic("123", "proj.dset", mock_st, mock_read_from_bq) # Missing table_id
            mock_st.error.assert_called_with("Invalid BigQuery Table Path format. Expected 'project.dataset.table'.")
            mock_read_from_bq.assert_not_called()
        self._run_test_with_patches(logic)


    def test_main_ui_update_workflow_success_single_fhrsid(self):
        fhrsid = "999"
        bq_path = "proj.dset.tbl"
        new_review_value = "Looks good"
        
        self.current_mock_session_state.update({
            'fhrsid_df': pd.DataFrame({'fhrsid': [fhrsid], 'manual_review': ['old_value']}),
            'successful_fhrsids': [fhrsid],
            'fhrsid_input_str_ui': fhrsid,
            'bq_table_lookup_input_str_ui': bq_path
        })

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
                if "Enter FHRSIDs" in label: return self.current_mock_session_state['fhrsid_input_str_ui']
                if "Enter BigQuery Table Path" in label: return self.current_mock_session_state['bq_table_lookup_input_str_ui']
                return value if value is not None else ""
            mock_st.text_input.side_effect = text_input_side_effect
            mock_st.multiselect.return_value = self.current_mock_session_state['successful_fhrsids']
            # If only one FHRSID, selectbox is not called for FHRSID selection.
            # mock_st.selectbox implicitly won't be called or its call won't matter.

        def logic(mock_st, mock_read_from_bq, mock_update_review): # mock_pd_concat removed
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
            # The call to read_from_bigquery during refresh should use STRING fhsrids
            mock_read_from_bq.assert_called_with([fhrsid], "proj", "dset", "tbl") # Changed int(fhrsid) to fhrsid
            mock_st.success.assert_any_call(f"Manual review updated for FHRSIDs: {fhrsid}. Refreshing data...")
            mock_st.rerun.assert_called_once()
            self.assertEqual(self.current_mock_session_state['fhrsid_df']['manual_review'].iloc[0], new_review_value)

        self._run_test_with_patches(logic, mock_st_config)

    def test_main_ui_update_workflow_update_fails(self):
        fhrsid = "1001"
        bq_path = "proj.dset.tbl"
        new_review_value = "This will fail"

        self.current_mock_session_state.update({
            'fhrsid_df': pd.DataFrame({'fhrsid': [fhrsid], 'manual_review': ['initial_state']}),
            'successful_fhrsids': [fhrsid],
            'fhrsid_input_str_ui': fhrsid,
            'bq_table_lookup_input_str_ui': bq_path
        })

        def mock_st_config(mock_st):
            mock_st.radio.return_value = "FHRSID Lookup"
            button_return_values = {"Update Manual Review": True, "Lookup FHRSIDs": False}
            # Revert to simpler lambda
            mock_st.button.side_effect = lambda text, key=None: button_return_values.get(text, False)

            def text_input_side_effect(label, value=None, key=None):
                if "New Manual Review Value" in label: return new_review_value
                if "Enter FHRSIDs" in label: return self.current_mock_session_state['fhrsid_input_str_ui']
                if "Enter BigQuery Table Path" in label: return self.current_mock_session_state['bq_table_lookup_input_str_ui']
                return value if value is not None else ""
            mock_st.text_input.side_effect = text_input_side_effect
            mock_st.multiselect.return_value = self.current_mock_session_state['successful_fhrsids']

        def logic(mock_st, mock_read_from_bq, mock_update_review): # mock_pd_concat removed
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
