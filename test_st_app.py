import unittest
import re
from unittest.mock import patch, MagicMock, PropertyMock, call
import pandas as pd
import streamlit as st # We will mock this heavily

# Import functions from st_app.py
# Assuming st_app.py is in the same directory or accessible via PYTHONPATH
from st_app import fhrsid_lookup_logic, main_ui, handle_fetch_data_action
# We also need to patch functions imported by st_app, like read_from_bigquery and update_manual_review
# Also, BigQueryExecutionError may be raised by read_from_bigquery
from bq_utils import BigQueryExecutionError
# Import process_and_update_master_data as it's called by handle_fetch_data_action
# and we want to let it run (not mock it directly)
from data_processing import process_and_update_master_data

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
            # mock_read_from_bq.assert_called_with([fhrsid], "proj", "dset", "tbl") # Removed this assertion
            mock_st.success.assert_any_call(f"Manual review updated for FHRSIDs: {fhrsid} in BigQuery.") # Changed message
            mock_st.info.assert_any_call("Local data view updated. Use 'Lookup FHRSIDs' again if you need to refresh from BigQuery.") # Added this assertion
            # mock_st.rerun.assert_called_once() # Removed this assertion
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


@patch('st_app.st', autospec=True)
class TestHandleFetchDataAction(unittest.TestCase):

    def common_mocks(self, mock_st_global):
        """Helper to create a dictionary of common mocks for patch.multiple."""
        # Reset mocks for each test case if they are attributes of the test class
        self.mock_load_all_data_from_bq = MagicMock()
        self.mock_fetch_api_data = MagicMock()
        self.mock_upload_to_gcs = MagicMock()
        self.mock_write_to_bigquery_helper = MagicMock() # For _write_data_to_bigquery
        self.mock_display_data = MagicMock()
        # process_and_update_master_data will be imported and run directly.
        # load_master_data is effectively replaced by load_all_data_from_bq for BQ path

        return {
            'load_all_data_from_bq': self.mock_load_all_data_from_bq,
            'fetch_api_data': self.mock_fetch_api_data,
            'upload_to_gcs': self.mock_upload_to_gcs,
            '_write_data_to_bigquery': self.mock_write_to_bigquery_helper,
            'display_data': self.mock_display_data,
            # 'process_and_update_master_data': self.mock_process_and_update_master_data # Not mocking this
        }

    def test_successful_flow_with_data(self, mock_st_global):
        """Test a successful run with initial data, API data, GCS and BQ writes."""
        with patch.multiple('st_app', **self.common_mocks(mock_st_global)):
            # Configure mock return values
            initial_master_data = [{'FHRSID': 1, 'BusinessName': 'Old Cafe'}]
            api_establishments = [{'FHRSID': 2, 'BusinessName': 'New Cafe', 'Geocode.Longitude': '1.0', 'Geocode.Latitude': '1.0'}]

            self.mock_load_all_data_from_bq.return_value = initial_master_data
            self.mock_fetch_api_data.return_value = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}
            self.mock_upload_to_gcs.return_value = True # Simulate successful upload

            result = handle_fetch_data_action(
                coordinate_pairs_str="1.0,1.0", # Valid coordinates for _parse_coordinates
                max_results=100,
                gcs_destination_uri_str="gs://bucket/api_raw/",
                master_list_uri_str="proj.dset.master_table",
                gcs_master_output_uri_str="gs://bucket/master_out.json",
                bq_full_path_str="out_proj.out_dset.out_table"
            )

            self.mock_load_all_data_from_bq.assert_called_once_with("proj", "dset", "master_table")
            self.mock_fetch_api_data.assert_called_once() # Called once for the valid coordinate pair

            # Check GCS uploads
            self.assertEqual(self.mock_upload_to_gcs.call_count, 2)
            self.mock_upload_to_gcs.assert_any_call(data=unittest.mock.ANY, destination_uri=unittest.mock.ANY) # Check API raw upload
            self.mock_upload_to_gcs.assert_any_call(data=unittest.mock.ANY, destination_uri="gs://bucket/master_out.json") # Check master data upload

            self.mock_display_data.assert_called_once()
            self.mock_write_to_bigquery_helper.assert_called_once()

            # Verify data passed to _write_data_to_bigquery (simplified check)
            args, _ = self.mock_write_to_bigquery_helper.call_args
            written_data = args[0]
            self.assertEqual(len(written_data), 2) # Old Cafe + New Cafe
            self.assertTrue(any(d['BusinessName'] == 'New Cafe' for d in written_data))

            mock_st_global.success.assert_any_call("Total establishments fetched from all API calls: 1")
            # Check for the specific success message with regex
            expected_pattern = re.compile(r"Successfully uploaded combined raw API response to gs://bucket/api_raw/combined_api_response_.*\.json")
            found_gcs_api_success_message = False
            for call_args in mock_st_global.success.call_args_list:
                args, _ = call_args
                if args and isinstance(args[0], str) and expected_pattern.match(args[0]):
                    found_gcs_api_success_message = True
                    break
            self.assertTrue(found_gcs_api_success_message, "Expected st.success call with GCS API response upload message was not found.")
            mock_st_global.success.assert_any_call("Successfully uploaded master restaurant data to gs://bucket/master_out.json")

            self.assertEqual(len(result), 2)


    def test_load_all_data_from_bq_returns_empty(self, mock_st_global):
        """Test flow when master BQ table is empty or load_all_data_from_bq returns empty list."""
        with patch.multiple('st_app', **self.common_mocks(mock_st_global)):
            self.mock_load_all_data_from_bq.return_value = [] # Simulate empty master list from BQ
            api_establishments = [{'FHRSID': 1, 'BusinessName': 'First Cafe', 'Geocode.Longitude': '1.0', 'Geocode.Latitude': '1.0'}]
            self.mock_fetch_api_data.return_value = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}

            result = handle_fetch_data_action(
                coordinate_pairs_str="1.0,1.0",
                max_results=100,
                master_list_uri_str="proj.dset.empty_table",
                gcs_destination_uri_str="",
                gcs_master_output_uri_str="",
                bq_full_path_str="out_proj.out_dset.out_table"
            )

            self.mock_load_all_data_from_bq.assert_called_once_with("proj", "dset", "empty_table")
            mock_st_global.warning.assert_any_call("No data loaded from BigQuery table proj.dset.empty_table, or table is empty. Proceeding as if with an empty master list.")

            self.mock_fetch_api_data.assert_called_once()
            self.mock_write_to_bigquery_helper.assert_called_once()
            args, _ = self.mock_write_to_bigquery_helper.call_args
            written_data = args[0]
            self.assertEqual(len(written_data), 1) # Only the new cafe
            self.assertEqual(written_data[0]['BusinessName'], 'First Cafe')
            self.assertEqual(len(result), 1)

    def test_invalid_master_bq_identifier_format(self, mock_st_global):
        """Test error handling for invalid master_list_uri_str format."""
        with patch.multiple('st_app', **self.common_mocks(mock_st_global)):
            invalid_uris = ["proj.dset", "invalid", "", "proj..table", ".dset.table"]
            for invalid_uri in invalid_uris:
                # Reset mocks for st.stop() and st.error() for each iteration
                mock_st_global.reset_mock() # Resets all sub-mocks of mock_st_global like .error, .stop

                handle_fetch_data_action(
                    coordinate_pairs_str="0,0", max_results=100,
                    master_list_uri_str=invalid_uri,
                    gcs_destination_uri_str="", gcs_master_output_uri_str="",
                    bq_full_path_str="out.proj.table"
                )

                if not invalid_uri:
                    mock_st_global.error.assert_any_call("Master Restaurant BigQuery Table identifier is missing.")
                elif len(invalid_uri.split('.')) != 3 or not all(p for p in invalid_uri.split('.')):
                    expected_pattern = re.compile(r"Invalid Master Restaurant BigQuery Table format")
                    found_error_message = False
                    for call_args in mock_st_global.error.call_args_list:
                        args, _ = call_args
                        if args and isinstance(args[0], str) and expected_pattern.search(args[0]): # Using search to find the substring
                            found_error_message = True
                            break
                    self.assertTrue(found_error_message, "Expected st.error call with invalid BQ table format message was not found.")

                mock_st_global.stop.assert_called()
                self.mock_fetch_api_data.assert_not_called() # Should stop before API call
                self.mock_load_all_data_from_bq.assert_not_called() # Should stop before BQ load call
                # Reset mocks for next iteration if they were part of self.common_mocks setup
                self.mock_fetch_api_data.reset_mock()
                self.mock_load_all_data_from_bq.reset_mock()


    def test_api_fetches_no_new_establishments(self, mock_st_global):
        """Test handling when API returns no new establishments."""
        with patch.multiple('st_app', **self.common_mocks(mock_st_global)):
            initial_master_data = [{'FHRSID': 1, 'BusinessName': 'Existing Cafe', 'first_seen': '2023-01-01'}]
            self.mock_load_all_data_from_bq.return_value = initial_master_data
            self.mock_fetch_api_data.return_value = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}} # No new data

            result = handle_fetch_data_action(
                coordinate_pairs_str="0,0", max_results=100,
                master_list_uri_str="proj.dset.master_table",
                gcs_destination_uri_str="", gcs_master_output_uri_str="",
                bq_full_path_str="out_proj.out_dset.out_table"
            )

            self.mock_load_all_data_from_bq.assert_called_once()
            self.mock_fetch_api_data.assert_called_once()
            mock_st_global.info.assert_any_call("No establishments found from any of the API calls. Nothing to process further.")
            mock_st_global.stop.assert_called() # Expect st.stop if no API data

            # Because of st.stop(), these should not be called if no API data
            self.mock_write_to_bigquery_helper.assert_not_called()
            # result should be the return from st.stop() or whatever the state is before it.
            # Given st.stop() is called, the return value of handle_fetch_data_action itself is not standard.
            # The primary check is that processing stops.
            # Depending on st.stop() implementation in tests, result might be None or a special mock object.
            # For this test, let's assume st.stop effectively halts execution, so no further assertions on 'result' length.
