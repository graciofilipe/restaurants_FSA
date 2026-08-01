import datetime
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd
from app.services.api_client import fetch_api_data
from app.services.bq_utils import ORIGINAL_COLUMNS_TO_KEEP

def parse_coordinates(coordinate_pairs_str: str) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Parses newline-separated coordinate pairs (lon, lat)."""
    valid_coords, errors = [], []
    for i, line in enumerate(coordinate_pairs_str.strip().split('\n')):
        line = line.strip()
        if not line:
            continue
        try:
            lon, lat = line.split(',')
            valid_coords.append((float(lon.strip()), float(lat.strip())))
        except ValueError:
            errors.append(f"Error parsing coordinate line {i+1}: '{line}'.")
    return valid_coords, errors

def fetch_data_for_all_coordinates(valid_coords: List[Tuple[float, float]], max_results: int) -> List[Dict[str, Any]]:
    """Fetches and aggregates API data for coordinates."""
    all_establishments = []
    for lon, lat in valid_coords:
        page = 1
        while True:
            resp = fetch_api_data(lon, lat, max_results, page)
            time.sleep(1)
            if not resp:
                break
            ests = resp.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', []) or []
            all_establishments.extend(ests)
            if len(ests) < max_results:
                break
            page += 1
    return all_establishments

def load_master_data(
    project_id: str, dataset_id: str, table_id: str,
    load_bq_func: Callable[[str, str, str], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Loads master restaurant data from BigQuery."""
    data = load_bq_func(project_id, dataset_id, table_id)
    if data is None:
        return []
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and r.get("manual_review") is None:
                r["manual_review"] = "not reviewed"
        return data
    raise TypeError(f"Expected list format from {project_id}.{dataset_id}.{table_id}, found {type(data)}.")

def process_and_update_master_data(
    master_data: List[Dict[str, Any]], api_data: Dict[str, Any], today_date: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Processes API data to identify newly added establishments."""
    today_date = today_date or datetime.datetime.now().strftime("%Y-%m-%d")
    raw_ests = api_data.get('FHRSEstablishment', {}).get('EstablishmentCollection', {}).get('EstablishmentDetail', [])
    messages = []
    if raw_ests is None:
        api_ests = []
        messages.append("No 'EstablishmentDetail' found in API response or it was None. No new establishments from API to process.")
    elif not raw_ests:
        api_ests = []
        messages.append("API response contained no establishments in 'EstablishmentDetail'.")
    else:
        api_ests = raw_ests

    existing_ids = set()
    for est in master_data:
        if isinstance(est, dict):
            fid = est.get('FHRSID') or est.get('fhrsid')
            if fid is not None:
                try:
                    existing_ids.add(str(int(fid)))
                except (ValueError, TypeError):
                    existing_ids.add(str(fid).strip().lower())

    new_records = []
    processed_ids = set()
    for est in api_ests:
        if isinstance(est, dict) and est.get('FHRSID') is not None:
            raw_id = est['FHRSID']
            try:
                cid = str(int(raw_id))
            except (ValueError, TypeError):
                cid = str(raw_id).strip().lower()

            est['FHRSID'] = cid
            if cid not in existing_ids and cid not in processed_ids:
                est['first_seen'] = today_date
                est['manual_review'] = "not reviewed"
                new_records.append({k: est.get(k) for k in ORIGINAL_COLUMNS_TO_KEEP})
                processed_ids.add(cid)

    count = len(new_records)
    if count > 0:
        summary_msg = f"Processed API response. Identified {count} unique new restaurant records to be added."
    elif messages:
        summary_msg = " ".join(messages)
    else:
        summary_msg = "Processed API response. No new restaurant records identified (or all were duplicates within the batch or already in BigQuery)."
    return new_records, summary_msg

def parse_bq_path(bq_path: str) -> Tuple[str, str, str]:
    """Parses 'project.dataset.table' format."""
    parts = bq_path.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid BigQuery Path: '{bq_path}'. Expected format: 'project.dataset.table'")
    return parts[0], parts[1], parts[2]

def parse_insight_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parses structured V2 or legacy text V1 insight columns."""
    res = {
        "insight_score": None, "insight_authenticity": None, "insight_verdict": "PENDING",
        "insight_summary": None, "insight_vibe": None, "detailed_insights": None
    }
    v2_json = row.get('gemini_insights_structured')
    if v2_json:
        try:
            d = json.loads(v2_json)
            res["detailed_insights"] = d
            res["insight_score"] = d.get('match_score')
            res["insight_authenticity"] = d.get('cultural_authenticity_rating')
            res["insight_summary"] = d.get('summary_reasoning')
            res["insight_vibe"] = d.get('atmosphere')

            score = d.get('match_score', 0)
            res["insight_verdict"] = "ACCEPTED" if score >= 85 else ("MAYBE" if score >= 70 else "REJECTED")

            res["match_score"] = d.get("match_score")
            res["1_value_and_volume_rating"] = d.get("1_value_and_volume", {}).get("rating")
            res["1_value_and_volume_verdict"] = d.get("1_value_and_volume", {}).get("verdict")
            res["2_demographic_community_score"] = d.get("2_demographic_community", {}).get("score")
            res["2_demographic_community_evidence"] = d.get("2_demographic_community", {}).get("evidence")
            res["3_linguistic_signal_score"] = d.get("3_linguistic_signal", {}).get("score")
            res["3_linguistic_signal_menu_type"] = d.get("3_linguistic_signal", {}).get("menu_type")
            res["4_geographic_precision_region_identified"] = d.get("4_geographic_precision", {}).get("region_identified")
            res["4_geographic_precision_specificity_level"] = d.get("4_geographic_precision", {}).get("specificity_level")
            res["5_culinary_uncompromisingness_score"] = d.get("5_culinary_uncompromisingness", {}).get("score")
            res["5_culinary_uncompromisingness_pander_check"] = d.get("5_culinary_uncompromisingness", {}).get("pander_check")
            res["6_establishment_integrity_is_sit_down_restaurant"] = d.get("6_establishment_integrity", {}).get("is_sit_down_restaurant")
            res["6_establishment_integrity_type"] = d.get("6_establishment_integrity", {}).get("type")
            res["summary_reasoning"] = d.get("summary_reasoning")
            return res
        except (json.JSONDecodeError, TypeError):
            pass

    v1_text = row.get('gemini_insights')
    if isinstance(v1_text, str) and v1_text.strip():
        res["insight_summary"] = v1_text
        up = v1_text.upper()
        if "FINAL VERDICT: ACCEPTED" in up:
            res["insight_verdict"], res["insight_score"] = "ACCEPTED", 90
        elif "FINAL VERDICT: PROBABLY REJECTED" in up or "FINAL VERDICT: REJECTED" in up:
            res["insight_verdict"], res["insight_score"] = "REJECTED", 20
        elif "FINAL VERDICT: MAYBE" in up or "UNSURE" in up:
            res["insight_verdict"], res["insight_score"] = "MAYBE", 50
    return res

def enhance_dataframe_with_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches DataFrame with parsed insight columns."""
    if df.empty:
        return df
    parsed = df.apply(lambda row: pd.Series(parse_insight_row(row)), axis=1)
    return pd.concat([df, parsed], axis=1)
