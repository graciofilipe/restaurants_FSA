import json
import time
import pandas as pd
import datetime
from typing import List, Dict, Any, Callable, Tuple, Optional
from app.services.bq_utils import ORIGINAL_COLUMNS_TO_KEEP
from app.services.api_client import fetch_api_data

def parse_coordinates(coordinate_pairs_str: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """
    Parses a string of coordinate pairs (lon, lat) separated by newlines.
    Returns a tuple containing:
    1. List of valid coordinate tuples (float, float).
    2. List of error messages for invalid lines.
    """
    valid_coords = []
    errors = []
    coordinate_lines = coordinate_pairs_str.strip().split('\n')
    for i, line in enumerate(coordinate_lines):
        line = line.strip()
        if not line: continue
        try:
            lon_str, lat_str = line.split(',')
            valid_coords.append((float(lon_str.strip()), float(lat_str.strip())))
        except ValueError:
            errors.append(f"Error parsing coordinate line {i+1}: '{line}'.")
    return valid_coords, errors

def fetch_data_for_all_coordinates(valid_coords: List[Tuple[float, float]], max_results: int) -> List[Dict[str, Any]]:
    """
    Fetches data from the API for all provided coordinates.
    Aggregates results up to max_results per coordinate.
    """
    all_api_establishments = []
    for lon, lat in valid_coords:
        page = 1
        while True:
            api_response = fetch_api_data(lon, lat, max_results, page)
            time.sleep(1)
            if api_response:
                establishments = api_response.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
                if establishments is None: establishments = []
                all_api_establishments.extend(establishments)
                if len(establishments) < max_results: break
                page += 1
            else: break
    return all_api_establishments

def load_json_from_local_file_path(uri: str) -> Optional[Dict[str, Any]]:
    """
    Loads a JSON file from a local file path.

    Args:
        uri: The local file path of the JSON file (e.g., "/path/to/file.json").

    Returns:
        A dictionary loaded from the JSON file, or None if an error occurs.
    """
    try:
        with open(uri, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Caller should handle None or we could raise exception
        return None
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def load_master_data(project_id: str, dataset_id: str, table_id: str, load_bq_func: Callable[[str, str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Loads master restaurant data from a BigQuery table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.
        load_bq_func: The function to use for loading data from BigQuery.
                      Expected to be bq_utils.load_all_data_from_bq.

    Returns:
        A list of dictionaries representing the master restaurant data.
    
    Raises:
        Exception: If loading fails (propagates from load_bq_func).
    """
    # Note: Logic removed logging to UI (st.info/error). Caller must handle logging.
    
    loaded_data = load_bq_func(project_id, dataset_id, table_id)

    if loaded_data is None: 
        # Previously returned [] with a warning. Now we return [] but maybe caller handles warning?
        # Or let's just return [] as before but without side effect.
        return []
    
    if isinstance(loaded_data, list):
        # Retain existing logic for default 'manual_review'
        for restaurant in loaded_data:
            if isinstance(restaurant, dict) and restaurant.get("manual_review") is None:
                restaurant["manual_review"] = "not reviewed"
        return loaded_data
    else:
        # Data format issue
        raise TypeError(f"Data loaded from BigQuery table {project_id}.{dataset_id}.{table_id} is not in the expected list format. Type found: {type(loaded_data)}.")

def process_and_update_master_data(
    master_data: List[Dict[str, Any]], 
    api_data: Dict[str, Any],
    today_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Processes API data and identifies new establishments not present in the master data.

    Args:
        master_data: The current list of master restaurant data.
        api_data: The raw JSON data (as a dict) from the API.
        today_date: Optional date string (YYYY-MM-DD) to use as 'first_seen'. Defaults to current date.

    Returns:
        A tuple containing:
        1. A list of newly added restaurant dictionaries.
        2. A summary message string suitable for display.
    """
    if today_date is None:
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")

    api_establishments = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    
    messages = []
    
    if api_establishments is None: 
        api_establishments = []
        messages.append("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not api_establishments: 
         messages.append("API response contained no establishments in 'EstablishmentDetail'.")

    existing_fhrsid_set = set()
    for est in master_data:
        if isinstance(est, dict):
            fhrsid_val = None
            if 'FHRSID' in est and est['FHRSID'] is not None:
                fhrsid_val = est['FHRSID']
            elif 'fhrsid' in est and est['fhrsid'] is not None:
                fhrsid_val = est['fhrsid']

            if fhrsid_val is not None:
                try:
                    canonical_fhrsid = str(int(fhrsid_val))
                except (ValueError, TypeError):
                    canonical_fhrsid = str(fhrsid_val).strip().lower()
                    # Warning suppressed or could be collected if needed
                existing_fhrsid_set.add(canonical_fhrsid)

    newly_added_restaurants: List[Dict[str, Any]] = []
    fhrsids_processed_in_this_batch = set() 

    for api_establishment in api_establishments:
        if isinstance(api_establishment, dict) and 'FHRSID' in api_establishment and api_establishment['FHRSID'] is not None:
            original_api_fhrsid = api_establishment['FHRSID']
            try:
                canonical_api_fhrsid = str(int(original_api_fhrsid))
            except ValueError:
                canonical_api_fhrsid = str(original_api_fhrsid).strip().lower()

            api_establishment['FHRSID'] = canonical_api_fhrsid

            if canonical_api_fhrsid not in existing_fhrsid_set:
                if canonical_api_fhrsid not in fhrsids_processed_in_this_batch:
                    api_establishment['first_seen'] = today_date
                    api_establishment['manual_review'] = "not reviewed"

                    processed_establishment = {}
                    for key in ORIGINAL_COLUMNS_TO_KEEP:
                        if key in api_establishment:
                            processed_establishment[key] = api_establishment[key]
                        else:
                            processed_establishment[key] = None

                    newly_added_restaurants.append(processed_establishment)
                    fhrsids_processed_in_this_batch.add(canonical_api_fhrsid) 

    count_new_restaurants = len(newly_added_restaurants)
    if count_new_restaurants > 0:
        summary_msg = f"Processed API response. Identified {count_new_restaurants} unique new restaurant records to be added."
    else:
        # Combine previous messages if any
        if messages:
            summary_msg = " ".join(messages)
        else:
            summary_msg = "Processed API response. No new restaurant records identified (or all were duplicates within the batch or already in BigQuery)."

    return newly_added_restaurants, summary_msg

def parse_bq_path(bq_path: str) -> Tuple[str, str, str]:
    """
    Parses a BigQuery path string in the format 'project.dataset.table'.
    """
    try:
        project_id, dataset_id, table_id = bq_path.split('.')
        return project_id, dataset_id, table_id
    except ValueError:
        raise ValueError(f"Invalid BigQuery Path: '{bq_path}'. Expected format: 'project.dataset.table'")

def parse_insight_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the insight columns (structured V2 and legacy V1) to extract standardized metrics.
    
    Returns a dict with:
    - insight_score: int (0-100)
    - insight_authenticity: int (1-5) (Legacy/Compat for filter)
    - insight_verdict: str (ACCEPTED, REJECTED, etc)
    - insight_summary: str
    - detailed_insights: dict (Full JSON for dynamic UI rendering)
    """
    # Default values
    result = {
        "insight_score": None,
        "insight_authenticity": None,
        "insight_verdict": "PENDING",
        "insight_summary": None,
        "insight_vibe": None,
        "detailed_insights": None
    }
    
    # 1. Try V2 (Structured)
    v2_json = row.get('gemini_insights_structured')
    if v2_json:
        try:
            data = json.loads(v2_json)
            result["detailed_insights"] = data
            result["insight_score"] = data.get('match_score')
            # Compat for filters if 'cultural_authenticity_rating' exists, else try to find it in new schema or leave None
            result["insight_authenticity"] = data.get('cultural_authenticity_rating') 
            result["insight_summary"] = data.get('summary_reasoning')
            result["insight_vibe"] = data.get('atmosphere')
            
            # Synthesize Verdict from Score
            score = data.get('match_score', 0)
            if score >= 85:
                result["insight_verdict"] = "ACCEPTED"
            elif score >= 70:
                result["insight_verdict"] = "MAYBE"
            else:
                result["insight_verdict"] = "REJECTED"

            # Flattened Columns for Display
            result["match_score"] = data.get("match_score")
            
            # 1. Value & Volume
            val_vol = data.get("1_value_and_volume", {})
            result["1_value_and_volume_rating"] = val_vol.get("rating")
            result["1_value_and_volume_verdict"] = val_vol.get("verdict")
            
            # 2. Demographic
            dem = data.get("2_demographic_community", {})
            result["2_demographic_community_score"] = dem.get("score")
            result["2_demographic_community_evidence"] = dem.get("evidence")
            
            # 3. Linguistic
            ling = data.get("3_linguistic_signal", {})
            result["3_linguistic_signal_score"] = ling.get("score")
            result["3_linguistic_signal_menu_type"] = ling.get("menu_type")
            
            # 4. Geographic
            geo = data.get("4_geographic_precision", {})
            result["4_geographic_precision_region_identified"] = geo.get("region_identified")
            result["4_geographic_precision_specificity_level"] = geo.get("specificity_level")
            
            # 5. Culinary
            cul = data.get("5_culinary_uncompromisingness", {})
            result["5_culinary_uncompromisingness_score"] = cul.get("score")
            result["5_culinary_uncompromisingness_pander_check"] = cul.get("pander_check")
            
            # 6. Integrity
            integ = data.get("6_establishment_integrity", {})
            result["6_establishment_integrity_is_sit_down_restaurant"] = integ.get("is_sit_down_restaurant")
            result["6_establishment_integrity_type"] = integ.get("type")
            
            result["summary_reasoning"] = data.get("summary_reasoning")
                
            return result
        except (json.JSONDecodeError, TypeError):
            pass # Fallback to V1
            
    # 2. Try V1 (Legacy Text)
    v1_text = row.get('gemini_insights')
    if v1_text:
        result["insight_summary"] = v1_text # Full text as summary
        upper_text = v1_text.upper()
        
        if "FINAL VERDICT: ACCEPTED" in upper_text:
            result["insight_verdict"] = "ACCEPTED"
            result["insight_score"] = 90 # Proxy score
        elif "FINAL VERDICT: PROBABLY REJECTED" in upper_text or "FINAL VERDICT: REJECTED" in upper_text:
             result["insight_verdict"] = "REJECTED"
             result["insight_score"] = 20
        elif "FINAL VERDICT: MAYBE" in upper_text or "UNSURE" in upper_text:
            result["insight_verdict"] = "MAYBE"
            result["insight_score"] = 50
    
    return result

def enhance_dataframe_with_insights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the dataframe with parsed insight columns.
    """
    if df.empty:
        return df
        
    # Apply parsing row by row
    parsed_data = df.apply(lambda row: pd.Series(parse_insight_row(row)), axis=1)
    
    # Concatenate with original DF
    # We use join (index based)
    df_enriched = pd.concat([df, parsed_data], axis=1)
    return df_enriched
