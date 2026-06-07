import os
import requests
import time
from google.cloud import bigquery

DEFAULT_BQ_PATH = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"
BQ_PATH = os.environ.get("BQ_PATH", DEFAULT_BQ_PATH)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

def main():
    if not API_KEY:
        print("Error: GOOGLE_MAPS_API_KEY is not set.")
        return
    
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    query = f"""
        SELECT fhrsid, BusinessName, PostCode, AddressLine1
        FROM `{table_ref}`
        WHERE maps_rating IS NULL AND maps_reviews IS NULL AND price_level IS NULL AND BusinessName IS NOT NULL
        LIMIT 3000
    """
    
    print(f"Fetching restaurants to enrich from {table_ref}...")
    try:
        results = client.query(query).result()
    except Exception as e:
        print(f"Error fetching from BQ: {e}")
        return
        
    rows_to_update = list(results)
    print(f"Found {len(rows_to_update)} restaurants to enrich.")
    
    if not rows_to_update:
        return

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.priceLevel,places.rating,places.userRatingCount"
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
                        "maps_reviews": maps_reviews
                    })
                    print(f"Enriched {name} -> rating: {maps_rating}, reviews: {maps_reviews}")
                else:
                    print(f"Place not found for {name}")
                    updates.append({
                        "fhrsid": row.fhrsid,
                        "price_level": None,
                        "maps_rating": -1.0,
                        "maps_reviews": -1
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
                values_list.append(f"('{fhrsid}', {pl}, {mr}, {mrev})")
                
            values_str = ",\n".join(values_list)
            
            merge_query = f"""
            MERGE `{table_ref}` T
            USING (
                SELECT * FROM UNNEST([
                    STRUCT<fhrsid STRING, price_level INT64, maps_rating FLOAT64, maps_reviews INT64>
                    {values_str}
                ])
            ) S
            ON T.fhrsid = S.fhrsid
            WHEN MATCHED THEN
              UPDATE SET 
                price_level = S.price_level,
                maps_rating = S.maps_rating,
                maps_reviews = S.maps_reviews
            """
            
            try:
                client.query(merge_query).result()
                print(f"Successfully updated batch of {len(batch)} rows.")
            except Exception as e:
                print(f"Merge error on batch: {e}")

if __name__ == "__main__":
    main()
