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

import math
import os
import re

# Load comprehensive UK / Greater London outcode centroids
_OUTCODES_JSON_PATH = os.path.join(os.path.dirname(__file__), "london_outcodes.json")
LONDON_OUTCODE_CENTROIDS: Dict[str, Tuple[float, float]] = {}

if os.path.exists(_OUTCODES_JSON_PATH):
    try:
        with open(_OUTCODES_JSON_PATH, "r", encoding="utf-8") as _f:
            _raw_coords = json.load(_f)
            LONDON_OUTCODE_CENTROIDS = {k.upper(): (float(v[0]), float(v[1])) for k, v in _raw_coords.items()}
    except Exception as _e:
        pass

# Core fallbacks if json missing
if not LONDON_OUTCODE_CENTROIDS:
    LONDON_OUTCODE_CENTROIDS = {
        "SW16": (51.4277, -0.1294),
        "SW2": (51.4500, -0.1200),
        "SW4": (51.4600, -0.1400),
        "SW8": (51.4750, -0.1300),
        "SW9": (51.4650, -0.1150),
        "SW11": (51.4650, -0.1650),
        "SW12": (51.4450, -0.1500),
        "SW17": (51.4300, -0.1650),
        "SW19": (51.4200, -0.2050),
        "SE1": (51.4990, -0.0900),
        "SE5": (51.4700, -0.0900),
        "SE11": (51.4880, -0.1100),
        "SE15": (51.4700, -0.0650),
        "SE24": (51.4550, -0.1000),
        "SE27": (51.4350, -0.1050),
        "EC1": (51.5230, -0.0980),
        "EC2": (51.5180, -0.0850),
        "WC1": (51.5220, -0.1220),
        "WC2": (51.5120, -0.1240),
        "W1": (51.5150, -0.1420),
        "W2": (51.5160, -0.1780),
        "N1": (51.5380, -0.1020),
        "E1": (51.5150, -0.0600),
        "E2": (51.5300, -0.0600),
        "E8": (51.5450, -0.0750),
    }

def extract_outcode(postcode_str: str) -> str:
    """Extracts the UK outcode from a postcode string (e.g. 'SW4 7UL' -> 'SW4', 'SW196NW' -> 'SW19')."""
    if not postcode_str or pd.isna(postcode_str):
        return "SW16"
    s = str(postcode_str).strip().upper()
    if ' ' in s:
        return s.split(' ')[0].strip()
    clean = re.sub(r'[^A-Z0-9]', '', s)
    if len(clean) >= 5 and re.match(r'^[A-Z0-9]+[0-9][A-Z]{2}$', clean):
        return clean[:-3]
    return clean

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates the great-circle distance between two points in kilometers."""
    try:
        lat1, lon1, lat2, lon2 = float(lat1), float(lon1), float(lat2), float(lon2)
    except (ValueError, TypeError):
        return 5.0
    r = 6371.0 # Earth's radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(r * c, 2)

def get_outcode_coordinates(outcode: str) -> Tuple[float, float]:
    """Retrieves centroid coordinates (lat, lon) for a UK outcode or postcode."""
    sw16_c = LONDON_OUTCODE_CENTROIDS.get("SW16", (51.4212, -0.1292))
    if not outcode or pd.isna(outcode):
        return sw16_c
    
    clean_oc = extract_outcode(outcode)
    if clean_oc in LONDON_OUTCODE_CENTROIDS:
        return tuple(LONDON_OUTCODE_CENTROIDS[clean_oc])
    
    # Try longest prefix match (e.g. EC2A -> EC2, SW1A -> SW1)
    for k in sorted(LONDON_OUTCODE_CENTROIDS.keys(), key=len, reverse=True):
        if clean_oc.startswith(k):
            return tuple(LONDON_OUTCODE_CENTROIDS[k])
            
    # Default fallback to Central London (Trafalgar Square)
    return (51.5074, -0.1278)

def calculate_restaurant_priority(
    df: pd.DataFrame,
    anchor_lat: Optional[float] = None,
    anchor_lon: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    today_date: Optional[datetime.date] = None
) -> pd.DataFrame:
    """
    Computes distance, proximity score, staleness score, Google Maps prior, and composite priority score.
    Returns the DataFrame augmented with 'distance_km' and 'priority_score'.
    """
    if df is None or df.empty:
        return df

    sw16_c = LONDON_OUTCODE_CENTROIDS.get("SW16", (51.4212, -0.1292))
    try:
        anchor_lat = float(anchor_lat) if anchor_lat is not None else sw16_c[0]
        anchor_lon = float(anchor_lon) if anchor_lon is not None else sw16_c[1]
    except (ValueError, TypeError):
        anchor_lat, anchor_lon = sw16_c

    weights = weights or {"prox": 0.35, "stale": 0.35, "prior": 0.20, "scope": 0.10}
    w_prox = weights.get("prox", 0.35)
    w_stale = weights.get("stale", 0.35)
    w_prior = weights.get("prior", 0.20)
    w_scope = weights.get("scope", 0.10)
    total_w = w_prox + w_stale + w_prior + w_scope
    if total_w > 0:
        w_prox, w_stale, w_prior, w_scope = w_prox / total_w, w_stale / total_w, w_prior / total_w, w_scope / total_w

    curr_date = today_date or datetime.date.today()
    res_df = df.copy()

    distances = []
    prox_scores = []
    stale_scores = []
    prior_scores = []
    scope_scores = []
    priority_scores = []

    for _, row in res_df.iterrows():
        # 1. Proximity & Distance
        lat = row.get('latitude')
        lon = row.get('longitude')
        has_exact = False
        if pd.notna(lat) and pd.notna(lon):
            try:
                lat_f, lon_f = float(lat), float(lon)
                if 45.0 <= lat_f <= 60.0 and -10.0 <= lon_f <= 5.0:
                    dist = haversine_distance_km(lat_f, lon_f, anchor_lat, anchor_lon)
                    has_exact = True
            except (ValueError, TypeError):
                has_exact = False

        if not has_exact:
            pc = row.get('postcode') or row.get('PostCode') or ""
            c_lat, c_lon = get_outcode_coordinates(str(pc))
            dist = haversine_distance_km(c_lat, c_lon, anchor_lat, anchor_lon)

        distances.append(dist)
        s_prox = round(100.0 * math.exp(-0.20 * dist), 1)
        prox_scores.append(s_prox)

        # 2. Staleness & Re-scoring (100 for unscored, 80 for >=60d, 60 for >=30d, 40 for >=14d, 15 for recent)
        pred_val = row.get('predicted_user_rating')
        gemini_val = row.get('gemini_insights') or row.get('gemini_insights_structured')
        if pd.isna(pred_val) or pd.isna(gemini_val) or pred_val is None or gemini_val is None:
            s_stale = 100.0
        else:
            score_ts = row.get('predicted_at') if pd.notna(row.get('predicted_at')) else row.get('first_seen')
            days_ago = 45 # Default medium staleness
            if score_ts and pd.notna(score_ts):
                try:
                    if isinstance(score_ts, str):
                        score_date = datetime.datetime.strptime(score_ts[:10], "%Y-%m-%d").date()
                    elif isinstance(score_ts, (datetime.date, datetime.datetime, pd.Timestamp)):
                        score_date = score_ts.date() if hasattr(score_ts, 'date') else score_ts
                    else:
                        score_date = curr_date
                    days_ago = max(0, (curr_date - score_date).days)
                except Exception:
                    days_ago = 45

            if days_ago >= 60:
                s_stale = 80.0
            elif days_ago >= 30:
                s_stale = 60.0
            elif days_ago >= 14:
                s_stale = 40.0
            else:
                s_stale = 15.0
        stale_scores.append(s_stale)

        # 3. Google Maps Quality Prior (FSA excluded)
        mr = row.get('maps_rating')
        if pd.notna(mr):
            try:
                base_mr = max(0.0, (float(mr) - 3.0) * 50.0)
                mrev = row.get('maps_reviews')
                rev_num = float(mrev) if pd.notna(mrev) else 0.0
                boost = min(15.0, math.log10(max(1.0, rev_num + 1.0)) * 5.0)
                s_prior = round(min(100.0, base_mr + boost), 1)
            except Exception:
                s_prior = 50.0
        else:
            s_prior = 50.0
        prior_scores.append(s_prior)

        # 4. Scope Confidence
        in_sc = row.get('in_scope')
        is_out_of_scope = False
        if in_sc is True or in_sc == 1 or str(in_sc).lower() in ("true", "1"):
            s_scope = 100.0
        elif in_sc is False or in_sc == 0 or str(in_sc).lower() in ("false", "0"):
            s_scope = 0.0
            is_out_of_scope = True
        else:
            s_scope = 50.0
        scope_scores.append(s_scope)

        # Composite Priority Score
        if is_out_of_scope:
            p = 0.0
        else:
            p = round((w_prox * s_prox) + (w_stale * s_stale) + (w_prior * s_prior) + (w_scope * s_scope), 1)
            # If restaurant already has a human user_rating, lower priority for ML scoring
            user_rt = row.get('user_rating')
            if pd.notna(user_rt) and str(user_rt).strip() != "":
                p = round(p * 0.1, 1)

        priority_scores.append(p)

    res_df['distance_km'] = distances
    res_df['priority_score'] = priority_scores
    res_df['proximity_score'] = prox_scores
    res_df['staleness_score'] = stale_scores
    res_df['maps_prior_score'] = prior_scores

    return res_df
