import unittest # Changed from pytest to unittest for consistency with TestAppendToBigQuery
from unittest.mock import MagicMock, patch
from app.core.data_processing import process_and_update_master_data
from app.services.bq_utils import ORIGINAL_COLUMNS_TO_KEEP # Import ORIGINAL_COLUMNS_TO_KEEP
from datetime import datetime
import pandas as pd # Added for potential pd.NA usage if needed by tested functions directly
import io

# --- Tests for process_and_update_master_data (modified) ---
class TestProcessAndUpdateMasterData(unittest.TestCase):
    def test_no_new_restaurants(self):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [{'FHRSID': "1", 'name': 'A'}]}}} # FHRSID is string
        
        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        
        self.assertEqual(len(new_restaurants), 0)
        self.assertIn("No new restaurant records identified", message)

    def test_accepts_a_bare_set_of_existing_fhrsids(self):
        """The cron loads IDs only, so master_data arrives as a set of strings.

        Silently ignoring a non-dict here would make every restaurant look new
        and duplicate the whole table on the next ingest.
        """
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': "1", 'BusinessName': 'Already known'},
            {'FHRSID': "2", 'BusinessName': 'Genuinely new'},
        ]}}}

        new_restaurants, _ = process_and_update_master_data({"1"}, api_data)

        self.assertEqual([r['FHRSID'] for r in new_restaurants], ["2"])

    def test_bare_fhrsids_are_normalised_before_comparison(self):
        """BigQuery hands back the ID as a string; the API sends an int."""
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': 1, 'BusinessName': 'Already known'},
        ]}}}

        new_restaurants, _ = process_and_update_master_data({"1"}, api_data)

        self.assertEqual(new_restaurants, [])

    def test_the_fsa_coordinates_are_kept_at_ingest(self):
        """R4. The FSA gives us a lat/lon with every establishment and we were
        throwing it away, then paying Google Places to tell us where the
        restaurant is. `Geocode` is nested, so it cannot survive the flat
        `ORIGINAL_COLUMNS_TO_KEEP` copy without being flattened first."""
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': "1", 'BusinessName': 'Waakye Joint',
             'Geocode': {'Longitude': '-0.129071', 'Latitude': '51.4213'}},
        ]}}}

        new_restaurants, _ = process_and_update_master_data([], api_data)

        self.assertEqual(new_restaurants[0]['latitude'], 51.4213)
        self.assertEqual(new_restaurants[0]['longitude'], -0.129071)

    def test_ingest_coordinates_are_numbers_not_the_api_strings(self):
        """`latitude`/`longitude` are FLOAT64 in `MASTER_BQ_SCHEMA`; the API
        sends them quoted, and the weekly load would reject the strings."""
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': "1", 'Geocode': {'Longitude': '-0.1', 'Latitude': '51.5'}},
        ]}}}

        new_restaurants, _ = process_and_update_master_data([], api_data)

        self.assertIsInstance(new_restaurants[0]['latitude'], float)
        self.assertIsInstance(new_restaurants[0]['longitude'], float)

    def test_a_missing_or_unusable_geocode_leaves_the_coordinates_null(self):
        """Null is the honest answer, and it is what lets Places fill the gap
        later -- a 0.0 would read as a point in the Gulf of Guinea."""
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': "1"},
            {'FHRSID': "2", 'Geocode': None},
            {'FHRSID': "3", 'Geocode': {'Longitude': '', 'Latitude': ''}},
            {'FHRSID': "4", 'Geocode': {'Longitude': 'n/a', 'Latitude': 'n/a'}},
        ]}}}

        new_restaurants, _ = process_and_update_master_data([], api_data)

        for record in new_restaurants:
            self.assertIsNone(record['latitude'], record['FHRSID'])
            self.assertIsNone(record['longitude'], record['FHRSID'])

    def test_a_lowercase_geocode_is_read_too(self):
        """The FSA API sends `Geocode`/`Latitude`; both spellings are checked
        at every other boundary in this file, and the API has changed the
        casing of its keys before."""
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            {'FHRSID': "1", 'geocode': {'longitude': '-0.1', 'latitude': '51.5'}},
        ]}}}

        new_restaurants, _ = process_and_update_master_data([], api_data)

        self.assertEqual(new_restaurants[0]['latitude'], 51.5)
        self.assertEqual(new_restaurants[0]['longitude'], -0.1)

    def test_add_new_restaurants_and_fields_initialization(self):
                # Setup mock for datetime.now().strftime()
                mock_datetime_str = "2023-10-26"
            
                master_data = [{'FHRSID': "1", 'BusinessName': 'A'}] # Existing record
                # Define API data with one existing and two new restaurants
                # These can have extra fields not in ORIGINAL_COLUMNS_TO_KEEP
                api_restaurant_1_existing = {'FHRSID': "1", 'BusinessName': 'A_updated', 'RatingValue': "Awful"}
                api_restaurant_2_new = {
                    'FHRSID': "2", 'BusinessName': 'Cafe Terra', 'RatingValue': '5',
                    'AddressLine1': '123 Main St', 'PostCode': 'AB1 2CD',
                    'LocalAuthorityName': 'Test Council', 'NewRatingPending': 'false',
                    'Scores': {'Hygiene': 10}, 'Geocode': {'Latitude': '1.0'}, 'BusinessType': 'Cafe'
                }
                api_restaurant_3_new = { # Minimal data, missing some optional ORIGINAL_COLUMNS_TO_KEEP fields
                    'FHRSID': "3", 'BusinessName': 'Pizza Place', 'RatingValue': '4',
                    'NewRatingPending': 'True', # String true
                    # Missing AddressLine1, PostCode, LocalAuthorityName from ORIGINAL_COLUMNS_TO_KEEP
                    'RatingDate': "2023-01-01" # This field is not in ORIGINAL_COLUMNS_TO_KEEP
                }
            
                api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
                    api_restaurant_1_existing,
                    api_restaurant_2_new,
                    api_restaurant_3_new
                ]}}}
            
                new_restaurants, message = process_and_update_master_data(master_data, api_data, today_date=mock_datetime_str)
            
                self.assertEqual(len(new_restaurants), 2)
                self.assertIn("Identified 2 unique new restaurant", message)
            
                # Check properties of the new restaurants
                for r_new in new_restaurants:
                    self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
                    self.assertEqual(r_new['first_seen'], mock_datetime_str)
                    
                    if r_new['FHRSID'] == "2": # api_restaurant_2_new
                        self.assertEqual(r_new['BusinessName'], 'Cafe Terra')
                        self.assertEqual(r_new['RatingValue'], '5')
                        self.assertEqual(r_new['AddressLine1'], '123 Main St')
                        self.assertEqual(r_new['PostCode'], 'AB1 2CD')
                        self.assertEqual(r_new['LocalAuthorityName'], 'Test Council')
                        self.assertEqual(r_new['NewRatingPending'], 'false') # Kept as string from API
                        # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not in API mock for this item should be None
                        self.assertIsNone(r_new.get('AddressLine2'))
                        self.assertIsNone(r_new.get('AddressLine3'))
                    elif r_new['FHRSID'] == "3": # api_restaurant_3_new
                        self.assertEqual(r_new['BusinessName'], 'Pizza Place')
                        self.assertEqual(r_new['RatingValue'], '4')
                        self.assertEqual(r_new['NewRatingPending'], 'True') # Kept as string from API
                        # These were missing in API data, so should be None
                        self.assertIsNone(r_new.get('AddressLine1'))
                        self.assertIsNone(r_new.get('AddressLine2'))
                        self.assertIsNone(r_new.get('AddressLine3'))
                        self.assertIsNone(r_new.get('PostCode'))
                        self.assertIsNone(r_new.get('LocalAuthorityName'))
        
                    # Assert that fields NOT in ORIGINAL_COLUMNS_TO_KEEP are absent
                    self.assertNotIn('Scores', r_new)
                    self.assertNotIn('Geocode', r_new)
                    self.assertNotIn('BusinessType', r_new)
                    self.assertNotIn('RatingDate', r_new) # Example of a field not kept

    def test_empty_master_data_all_api_items_are_new(self):
        master_data = []
        # API data can have more fields than ORIGINAL_COLUMNS_TO_KEEP
        api_restaurant = {
            'FHRSID': "1", 'BusinessName': 'Solo Cafe', 'RatingValue': 'Excellent',
            'AddressLine1': 'Addr1', 'PostCode': 'PC', 'LocalAuthorityName': 'LA',
            'NewRatingPending': 'false',
            'ExtraField': 'This will be dropped'
        }
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_restaurant]}}}
    
        mock_date_str = "mock_date_value"
        new_restaurants, message = process_and_update_master_data(master_data, api_data, today_date=mock_date_str)
    
        self.assertEqual(len(new_restaurants), 1)
        self.assertIn("Identified 1 unique new restaurant", message)
    
        r_new = new_restaurants[0]
        
        self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
        self.assertEqual(r_new['FHRSID'], "1")
        self.assertEqual(r_new['BusinessName'], 'Solo Cafe')
        self.assertEqual(r_new['RatingValue'], 'Excellent') # Preserved as per ORIGINAL_COLUMNS_TO_KEEP
        self.assertEqual(r_new['AddressLine1'], 'Addr1')
        self.assertEqual(r_new['PostCode'], 'PC')
        self.assertEqual(r_new['LocalAuthorityName'], 'LA')
        self.assertEqual(r_new['NewRatingPending'], 'false') # Preserved as string
        self.assertEqual(r_new['first_seen'], mock_date_str)
        # Optional fields from ORIGINAL_COLUMNS_TO_KEEP not provided in API mock
        self.assertIsNone(r_new.get('AddressLine2'))
        self.assertIsNone(r_new.get('AddressLine3'))

        self.assertNotIn('ExtraField', r_new) # Check that extra field is dropped

    def test_empty_api_data_detail(self):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': []}}}
        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        self.assertIn("API response contained no establishments", message)


    def test_api_data_establishment_detail_is_none(self):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': None}}}
        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        self.assertIn("No 'EstablishmentDetail' found in API response or it was None", message)

    def test_api_data_missing_establishment_collection(self):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {'FHRSEstablishment': {}} # EstablishmentCollection is missing
        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        self.assertIn("API response contained no establishments", message)


    def test_api_data_missing_fhrestablishment_key(self):
        master_data = [{'FHRSID': "1", 'name': 'A'}] # FHRSID is string
        api_data = {} # FHRSEstablishment key is missing
        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 0)
        self.assertIn("API response contained no establishments", message)

        def test_fhrsid_is_string_after_processing_and_schema_adherence(self):
            """
            Test FHRSID is string after processing, and output adheres to ORIGINAL_COLUMNS_TO_KEEP.
            """
            mock_date_str = "2023-10-27"
            master_data = []
        
            api_est_int_fhrsid = {
                'FHRSID': 123, 'BusinessName': 'Testaurant Int',
                'RatingValue': 'Good', 'LocalAuthorityName': 'LA1', 'NewRatingPending': 'false',
                'ExtraInfo': 'will be dropped'
            }
            api_est_str_fhrsid = {
                'FHRSID': "456", 'BusinessName': 'Testaurant Str',
                'AddressLine1': 'Street', 'PostCode': 'PC',
                'RatingValue': 'Bad', 'LocalAuthorityName': 'LA2', 'NewRatingPending': 'TRUE',
                'AnotherExtra': 'also dropped'
            }
            api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_est_int_fhrsid, api_est_str_fhrsid]}}}
        
            new_restaurants, message = process_and_update_master_data(master_data, api_data, today_date=mock_date_str)
            self.assertEqual(len(new_restaurants), 2)
            self.assertIn("Identified 2 unique new restaurant", message)
        
            for r_new in new_restaurants:
                self.assertEqual(set(r_new.keys()), set(ORIGINAL_COLUMNS_TO_KEEP))
                self.assertIsInstance(r_new['FHRSID'], str)
                self.assertEqual(r_new['first_seen'], mock_date_str)

            self.assertNotIn('ExtraInfo', r_new)
            self.assertNotIn('AnotherExtra', r_new)

    @patch('app.core.data_processing.datetime')
    def test_duplicate_fhrsid_in_api_data_is_added_once(self, mock_datetime):
        mock_datetime_str = "2023-10-28"
        mock_datetime.now.return_value.strftime.return_value = mock_datetime_str

        master_data = [{'FHRSID': "1", 'BusinessName': 'Old Restaurant'}]

        api_restaurant_duplicate_1 = {
            'FHRSID': "789", 'BusinessName': 'Duplicate Cafe Batch 1',
            'RatingValue': '5', 'AddressLine1': 'Addr D1', 'PostCode': 'PC D1',
            'LocalAuthorityName': 'LA D1', 'NewRatingPending': 'false'
        }
        api_restaurant_duplicate_2 = {
            'FHRSID': "789", 'BusinessName': 'Duplicate Cafe Batch 2',
            'RatingValue': '5', 'AddressLine1': 'Addr D2', 'PostCode': 'PC D2',
            'LocalAuthorityName': 'LA D2', 'NewRatingPending': 'false'
        }
        api_restaurant_unique_new = {
            'FHRSID': "101", 'BusinessName': 'Unique New Place',
            'RatingValue': '4', 'AddressLine1': 'Addr U1', 'PostCode': 'PC U1',
            'LocalAuthorityName': 'LA U1', 'NewRatingPending': 'true'
        }

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [
            api_restaurant_duplicate_1,
            api_restaurant_unique_new,
            api_restaurant_duplicate_2
        ]}}}

        new_restaurants, message = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2, "Should identify 2 unique new restaurants.")

        result_fhrsids = {r['FHRSID'] for r in new_restaurants}
        self.assertIn("789", result_fhrsids, "FHRSID 789 should be in the results.")
        self.assertIn("101", result_fhrsids, "FHRSID 101 should be in the results.")
        self.assertIn("Identified 2 unique new restaurant", message)

    @patch('app.core.data_processing.datetime')
    def test_canonical_fhrsid_deduplication_and_non_numeric(self, mock_datetime):
        mock_datetime_str = "2023-11-15"
        mock_datetime.now.return_value.strftime.return_value = mock_datetime_str

        master_data = [
            {'FHRSID': 123, 'BusinessName': 'Integer Master', 'AddressLine1': 'Addr Master 1'},
            {'FHRSID': "456", 'BusinessName': 'Canonical String Master', 'AddressLine1': 'Addr Master 2'},
            {'FHRSID': "ABC", 'BusinessName': 'NonNumeric Master', 'AddressLine1': 'Addr Master 3'},
            {'FHRSID': "M1X", 'BusinessName': 'Malformed Master', 'AddressLine1': 'Addr Master 4'},
            {'FHRSID': None, 'BusinessName': 'None FHRSID Master', 'AddressLine1': 'Addr Master 5'}
        ]

        api_establishments = [
            {'FHRSID': "0123", 'BusinessName': 'Integer API Dup', 'AddressLine1': 'Addr API 1'},
            {'FHRSID': "456", 'BusinessName': 'Canonical String API Dup', 'AddressLine1': 'Addr API 2'},
            {'FHRSID': "ABC", 'BusinessName': 'NonNumeric API Dup', 'AddressLine1': 'Addr API 3'},
            {'FHRSID': "M1X", 'BusinessName': 'Malformed API Dup', 'AddressLine1': 'Addr API 4'},
            {'FHRSID': "0789", 'BusinessName': 'New Numeric Normalized', 'AddressLine1': 'Addr API 5'},
            {'FHRSID': "DEF", 'BusinessName': 'New NonNumeric', 'AddressLine1': 'Addr API 6'},
            {'FHRSID': "A2Y", 'BusinessName': 'New Malformed API', 'AddressLine1': 'Addr API 7'},
            {'FHRSID': None, 'BusinessName': 'None FHRSID API', 'AddressLine1': 'Addr API 8'}
        ]
        for est_api in api_establishments:
            if est_api['FHRSID'] is not None:
                est_api.update({key: None for key in ORIGINAL_COLUMNS_TO_KEEP if key not in est_api})

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}

        new_restaurants, message = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 3, "Should identify 3 new unique restaurants.")
        self.assertIn("Identified 3 unique new restaurant", message)

        # Check that FHRSIDs are canonicalized (string for numeric, lowercased string for non-numeric)
        added_fhrsids = sorted([r['FHRSID'] for r in new_restaurants])
        expected_fhrsids = sorted(["789", "a2y", "def"])
        self.assertEqual(added_fhrsids, expected_fhrsids, "FHRSIDs of new restaurants should be the canonical forms.")

    @patch('app.core.data_processing.datetime')
    def test_deduplication_with_corrected_fhrsid_key(self, mock_datetime):
        mock_date_str = "2024-01-01"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str

        master_data = [
            {'fhrsid': "123", 'BusinessName': 'BQ Cafe Old'},
            {'fhrsid': "ABC", 'BusinessName': 'BQ NonNumeric Old'}
        ]

        api_establishments = [
            {'FHRSID': "123", 'BusinessName': 'API Cafe Update', 'RatingValue': '3', 'NewRatingPending': 'false',
             'AddressLine1': 'Addr1', 'AddressLine2': None, 'AddressLine3': None, 'PostCode': 'PC1',
             'LocalAuthorityName': 'LA1'},
            {'FHRSID': "789", 'BusinessName': 'API Cafe New Numeric', 'RatingValue': '5', 'NewRatingPending': 'false',
             'AddressLine1': 'Addr2', 'AddressLine2': 'Suite B', 'AddressLine3': None, 'PostCode': 'PC2',
             'LocalAuthorityName': 'LA2'},
            {'FHRSID': "ABC", 'BusinessName': 'API NonNumeric Update', 'RatingValue': '2', 'NewRatingPending': 'true',
             'AddressLine1': 'Addr3', 'AddressLine2': None, 'AddressLine3': 'Old Town', 'PostCode': 'PC3',
             'LocalAuthorityName': 'LA3'},
            {'FHRSID': "XYZ", 'BusinessName': 'API Cafe New NonNumeric', 'RatingValue': '1', 'NewRatingPending': 'true',
             'AddressLine1': 'Addr4', 'AddressLine2': None, 'AddressLine3': None, 'PostCode': 'PC4',
             'LocalAuthorityName': 'LA4'}
        ]
        for est_api in api_establishments:
            for key in ORIGINAL_COLUMNS_TO_KEEP:
                if key not in est_api:
                    est_api[key] = None
            if 'FHRSID' not in est_api:
                est_api['FHRSID'] = None


        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': api_establishments}}}

        new_restaurants, message = process_and_update_master_data(master_data, api_data)

        self.assertEqual(len(new_restaurants), 2, "Should identify 2 new unique restaurants.")
        self.assertIn("Identified 2 unique new restaurant", message)

        added_fhrsids = sorted([r['FHRSID'] for r in new_restaurants])
        expected_fhrsids = sorted(["789", "xyz"])
        self.assertEqual(added_fhrsids, expected_fhrsids)

    @patch('app.core.data_processing.datetime')
    def test_maps_link_not_added(self, mock_datetime):
        """
        Test that 'Maps Link' is NOT added to processed restaurants.
        """
        mock_date_str = "2023-10-27"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str
        master_data = []

        api_est = {
            'FHRSID': 123, 'BusinessName': 'Testaurant',
            'AddressLine1': '123 Main St', 'PostCode': 'A1 1AA',
            'RatingValue': '5', 'LocalAuthorityName': 'LA', 'NewRatingPending': 'false'
        }
        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_est]}}}

        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        self.assertEqual(len(new_restaurants), 1)
        r_new = new_restaurants[0]
        self.assertNotIn('Maps Link', r_new)

    @patch('app.core.data_processing.datetime')
    def test_deduplication_edge_cases_non_numeric(self, mock_datetime):
        """
        Test that deduplication handles casing and whitespace differences in NON-NUMERIC FHRSID.
        """
        mock_date_str = "2023-10-27"
        mock_datetime.now.return_value.strftime.return_value = mock_date_str
        
        # Master data has 'AbC' and ' XyZ '
        master_data = [
            {'FHRSID': "AbC", 'BusinessName': 'Cafe Alpha'}, 
            {'FHRSID': " XyZ ", 'BusinessName': 'Cafe Beta'}
        ]

        # API data has 'abc' (lowercase) and 'XyZ' (stripped)
        # Both should be identified as existing and NOT added.
        api_est1 = {'FHRSID': "abc", 'BusinessName': 'Cafe Alpha API'} 
        api_est2 = {'FHRSID': "XyZ", 'BusinessName': 'Cafe Beta API'}
        
        # Also add a genuinely new one 'NewID'
        api_est3 = {'FHRSID': "NewID", 'BusinessName': 'Cafe Gamma'}

        api_data = {'FHRSEstablishment': {'EstablishmentCollection': {'EstablishmentDetail': [api_est1, api_est2, api_est3]}}}

        new_restaurants, message = process_and_update_master_data(master_data, api_data)
        
        # Should only contain 'NewID' (normalized to 'newid')
        self.assertEqual(len(new_restaurants), 1)
        self.assertEqual(new_restaurants[0]['FHRSID'], "newid")


if __name__ == '__main__':
    unittest.main()



from app.core.data_processing import enhance_dataframe_with_insights
from app.core.pillar_schema import ALL_COLUMNS
import math


class TestEnhanceDataframeWithInsights(unittest.TestCase):
    """The pillars arrive from BigQuery as typed columns now.

    This used to call `parse_insight_row` on every row of every Streamlit
    rerun, each call running `json.loads` over the raw profile and rebuilding a
    flat naming convention that existed nowhere else. The Phase 5 backfill and
    the Phase 6 dual-write made that work redundant; the columns are already
    there and already typed.
    """

    def test_the_typed_columns_pass_through_untouched(self):
        df = pd.DataFrame([{'fhrsid': '1', 'match_score': 88, 'pillar_value_rating': 7,
                            'pillar_geo_specificity': 'HYPER_REGIONAL'}])
        out = enhance_dataframe_with_insights(df)
        self.assertEqual(out.loc[0, 'match_score'], 88)
        self.assertEqual(out.loc[0, 'pillar_value_rating'], 7)
        self.assertEqual(out.loc[0, 'pillar_geo_specificity'], 'HYPER_REGIONAL')

    def test_every_pillar_column_exists_afterwards(self):
        """`DISPLAY_COLUMNS` names all fourteen; a frame that lacks one would
        leave the grid ordering a column that is not there."""
        out = enhance_dataframe_with_insights(pd.DataFrame([{'fhrsid': '1'}]))
        for column in ALL_COLUMNS:
            self.assertIn(column, out.columns)

    def test_a_missing_column_reads_as_missing_not_zero(self):
        out = enhance_dataframe_with_insights(pd.DataFrame([{'fhrsid': '1'}]))
        self.assertTrue(pd.isna(out.loc[0, 'match_score']))

    def test_the_raw_json_is_no_longer_parsed(self):
        """The blob stays as the audit trail, but nothing reads it per row. A
        profile whose typed columns were never backfilled stays empty rather
        than being re-derived on the fly."""
        df = pd.DataFrame([{'fhrsid': '1',
                            'gemini_insights_structured': '{"match_score": 91}'}])
        out = enhance_dataframe_with_insights(df)
        self.assertTrue(pd.isna(out.loc[0, 'match_score']))
        self.assertEqual(out.loc[0, 'gemini_insights_structured'], '{"match_score": 91}')

    def test_an_empty_frame_is_returned_unchanged(self):
        empty = pd.DataFrame()
        self.assertTrue(enhance_dataframe_with_insights(empty).empty)

    def test_it_does_not_mutate_the_frame_it_was_given(self):
        df = pd.DataFrame([{'fhrsid': '1'}])
        enhance_dataframe_with_insights(df)
        self.assertEqual(list(df.columns), ['fhrsid'])
