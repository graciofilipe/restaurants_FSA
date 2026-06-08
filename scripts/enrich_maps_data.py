import os
import requests
import time
from google.cloud import bigquery

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
BQ_PATH = os.environ.get("BQ_PATH", DEFAULT_BQ_PATH)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

from typing import List, Optional

def enrich_restaurants_by_fhrsid(fhrsids: Optional[List[str]] = None, limit: int = 1000, force_regen: bool = False) -> int:
    
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    fhrsid_filter = ""
    if fhrsids:
        escaped_ids = [str(fid).replace("'", "''") for fid in fhrsids]
        id_list_str = ", ".join([f"'{fid}'" for fid in escaped_ids])
        fhrsid_filter = f"AND fhrsid IN ({id_list_str})"

    null_filter = "" if force_regen else "AND maps_rating IS NULL AND maps_reviews IS NULL"

    query = f"""
        SELECT fhrsid, BusinessName, PostCode, AddressLine1
        FROM `{table_ref}`
        WHERE BusinessName IS NOT NULL
          {null_filter}
          {fhrsid_filter}
        LIMIT {limit}
    """
    
    print(f"Fetching restaurants to enrich from {table_ref}...")
    try:
        results = client.query(query).result()
    except Exception as e:
        print(f"Error fetching from BQ: {e}")
        return 0
        
    rows_to_update = list(results)
    print(f"Found {len(rows_to_update)} restaurants to enrich.")
    
    if not rows_to_update:
        return 0

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.priceLevel,places.rating,places.userRatingCount,places.location,places.googleMapsUri,places.businessStatus,places.types,places.websiteUri"
    }
    
    updates = []
    
    for row in rows_to_update:
        name = row.BusinessName
        postcode = row.PostCode
        address = row.AddressLine1
        
        search_query = f"{name} {postcode}" if postcode else f"{name} {address}"
        
        payload = {
            "textQuery": search_query
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if "places" in data and len(data["places"]) > 0:
                    place = data["places"][0]
                    price_level_raw = place.get("priceLevel", None)
                    maps_rating = place.get("rating", None)
                    maps_reviews = place.get("userRatingCount", None)
                    
                    location = place.get("location", {})
                    latitude = location.get("latitude", None)
                    longitude = location.get("longitude", None)

                    maps_url = place.get("googleMapsUri", None)
                    business_status = place.get("businessStatus", None)
                    website_url = place.get("websiteUri", None)

                    types_list = place.get("types", [])
                    maps_types = ",".join(types_list) if types_list else None

                    price_level = None
                    if price_level_raw:
                        pl_map = {
                            "PRICE_LEVEL_FREE": 0,
                            "PRICE_LEVEL_INEXPENSIVE": 1,
                            "PRICE_LEVEL_MODERATE": 2,
                            "PRICE_LEVEL_EXPENSIVE": 3,
                            "PRICE_LEVEL_VERY_EXPENSIVE": 4
                        }
                        if isinstance(price_level_raw, str) and price_level_raw in pl_map:
                            price_level = pl_map[price_level_raw]
                        elif isinstance(price_level_raw, int):
                            price_level = price_level_raw
                    
                    updates.append({
                        "fhrsid": row.fhrsid,
                        "price_level": price_level,
                        "maps_rating": maps_rating,
                        "maps_reviews": maps_reviews,
                        "latitude": latitude,
                        "longitude": longitude,
                        "maps_url": maps_url,
                        "business_status": business_status,
                        "website_url": website_url,
                        "maps_types": maps_types
                    })
                    print(f"Enriched {name} -> rating: {maps_rating}, reviews: {maps_reviews}")
                else:
                    print(f"Place not found for {name}")
                    updates.append({
                        "fhrsid": row.fhrsid,
                        "price_level": None,
                        "maps_rating": -1.0,
                        "maps_reviews": -1,
                        "latitude": None,
                        "longitude": None,
                        "maps_url": None,
                        "business_status": None,
                        "website_url": None,
                        "maps_types": None
                    })
            else:
                print(f"API Error for {name}: {response.status_code} {response.text}")
                
        except Exception as e:
            print(f"Error fetching for {name}: {e}")
            
        time.sleep(0.05)
        
    if updates:
        print(f"Executing batch update of {len(updates)} rows...")
        
        batch_size = 500
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            values_list = []
            for u in batch:
                fhrsid = u["fhrsid"].replace("'", "\\'")
                pl = str(u["price_level"]) if u["price_level"] is not None else "NULL"
                mr = str(u["maps_rating"]) if u["maps_rating"] is not None else "NULL"
                mrev = str(u["maps_reviews"]) if u["maps_reviews"] is not None else "NULL"
                
                lat = str(u["latitude"]) if u["latitude"] is not None else "NULL"
                lon = str(u["longitude"]) if u["longitude"] is not None else "NULL"

                # Safely escape strings for SQL
                murl_val = u['maps_url'].replace("'", "\\'") if u["maps_url"] is not None else ""
                bstat_val = u['business_status'].replace("'", "\\'") if u["business_status"] is not None else ""
                wurl_val = u['website_url'].replace("'", "\\'") if u["website_url"] is not None else ""
                mtypes_val = u['maps_types'].replace("'", "\\'") if u["maps_types"] is not None else ""

                murl = f"'{murl_val}'" if u["maps_url"] is not None else "NULL"
                bstat = f"'{bstat_val}'" if u["business_status"] is not None else "NULL"
                wurl = f"'{wurl_val}'" if u["website_url"] is not None else "NULL"
                mtypes = f"'{mtypes_val}'" if u["maps_types"] is not None else "NULL"

                values_list.append(f"('{fhrsid}', {pl}, {mr}, {mrev}, {lat}, {lon}, {murl}, {bstat}, {wurl}, {mtypes})")
                
            values_str = ",\n".join(values_list)
            
            merge_query = f"""
            MERGE `{table_ref}` T
            USING (
                SELECT * FROM UNNEST([
                    STRUCT<fhrsid STRING, price_level INT64, maps_rating FLOAT64, maps_reviews INT64, latitude FLOAT64, longitude FLOAT64, maps_url STRING, business_status STRING, website_url STRING, maps_types STRING>
                    {values_str}
                ])
            ) S
            ON T.fhrsid = S.fhrsid
            WHEN MATCHED THEN
              UPDATE SET 
                price_level = S.price_level,
                maps_rating = S.maps_rating,
                maps_reviews = S.maps_reviews,
                latitude = S.latitude,
                longitude = S.longitude,
                maps_url = S.maps_url,
                business_status = S.business_status,
                website_url = S.website_url,
                maps_types = S.maps_types
            """
            
            try:
                client.query(merge_query).result()
                print(f"Successfully updated batch of {len(batch)} rows.")
            except Exception as e:
                print(f"Merge error on batch: {e}")

def main():
    enrich_restaurants_by_fhrsid()

if __name__ == "__main__":
    main()
