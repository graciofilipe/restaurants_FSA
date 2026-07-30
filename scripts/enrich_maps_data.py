import os
import time
from typing import List, Optional
from google.cloud import bigquery
import requests

BQ_PATH = os.environ.get("BQ_PATH", "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master")
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

PL_MAP = {
    "PRICE_LEVEL_FREE": 0, "PRICE_LEVEL_INEXPENSIVE": 1, "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3, "PRICE_LEVEL_VERY_EXPENSIVE": 4
}

def enrich_restaurants_by_fhrsid(fhrsids: Optional[List[str]] = None, limit: int = 1000, force_regen: bool = False) -> int:
    project_id, dataset_id, table_id = BQ_PATH.split(".")
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    fhrsid_filter = f"AND fhrsid IN ({', '.join(f'\'{str(f).replace(chr(39), chr(39)+chr(39))}\'' for f in fhrsids)})" if fhrsids else ""
    null_filter = "" if force_regen else "AND maps_rating IS NULL AND maps_reviews IS NULL"

    query = f"SELECT fhrsid, BusinessName, PostCode, AddressLine1 FROM `{table_ref}` WHERE BusinessName IS NOT NULL {null_filter} {fhrsid_filter} LIMIT {limit}"
    try:
        rows_to_update = list(client.query(query).result())
    except Exception as e:
        print(f"Error fetching from BQ: {e}")
        return 0

    if not rows_to_update:
        return 0

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json", "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "places.priceLevel,places.rating,places.userRatingCount,places.location,places.googleMapsUri,places.businessStatus,places.types,places.websiteUri"
    }

    updates = []
    for row in rows_to_update:
        search_query = f"{row.BusinessName} {row.PostCode}" if row.PostCode else f"{row.BusinessName} {row.AddressLine1}"
        try:
            resp = requests.post(url, json={"textQuery": search_query}, headers=headers)
            if resp.status_code == 200 and "places" in resp.json() and resp.json()["places"]:
                p = resp.json()["places"][0]
                pr = p.get("priceLevel")
                pl = PL_MAP.get(pr, pr) if isinstance(pr, (str, int)) else None
                loc = p.get("location", {})
                updates.append({
                    "fhrsid": row.fhrsid, "price_level": pl, "maps_rating": p.get("rating"), "maps_reviews": p.get("userRatingCount"),
                    "latitude": loc.get("latitude"), "longitude": loc.get("longitude"), "maps_url": p.get("googleMapsUri"),
                    "business_status": p.get("businessStatus"), "website_url": p.get("websiteUri"),
                    "maps_types": ",".join(p.get("types", [])) if p.get("types") else None
                })
            else:
                updates.append({"fhrsid": row.fhrsid, "price_level": None, "maps_rating": -1.0, "maps_reviews": -1, "latitude": None, "longitude": None, "maps_url": None, "business_status": None, "website_url": None, "maps_types": None})
        except Exception as e:
            print(f"Error fetching for {row.BusinessName}: {e}")
        time.sleep(0.05)

    if updates:
        for i in range(0, len(updates), 500):
            batch = updates[i:i + 500]
            val_strs = []
            for u in batch:
                fid = u["fhrsid"].replace("'", "\\'")
                pl, mr, mrev = u["price_level"] or "NULL", u["maps_rating"] if u["maps_rating"] is not None else "NULL", u["maps_reviews"] if u["maps_reviews"] is not None else "NULL"
                lat, lon = u["latitude"] if u["latitude"] is not None else "NULL", u["longitude"] if u["longitude"] is not None else "NULL"
                murl = f"'{u['maps_url'].replace(chr(39), chr(92)+chr(39))}'" if u["maps_url"] else "NULL"
                bstat = f"'{u['business_status'].replace(chr(39), chr(92)+chr(39))}'" if u["business_status"] else "NULL"
                wurl = f"'{u['website_url'].replace(chr(39), chr(92)+chr(39))}'" if u["website_url"] else "NULL"
                mtypes = f"'{u['maps_types'].replace(chr(39), chr(92)+chr(39))}'" if u["maps_types"] else "NULL"
                val_strs.append(f"('{fid}', {pl}, {mr}, {mrev}, {lat}, {lon}, {murl}, {bstat}, {wurl}, {mtypes})")

            merge_q = f"""
            MERGE `{table_ref}` T
            USING (SELECT * FROM UNNEST([STRUCT<fhrsid STRING, price_level INT64, maps_rating FLOAT64, maps_reviews INT64, latitude FLOAT64, longitude FLOAT64, maps_url STRING, business_status STRING, website_url STRING, maps_types STRING> {", ".join(val_strs)}])) S
            ON T.fhrsid = S.fhrsid
            WHEN MATCHED THEN UPDATE SET price_level=S.price_level, maps_rating=S.maps_rating, maps_reviews=S.maps_reviews, latitude=S.latitude, longitude=S.longitude, maps_url=S.maps_url, business_status=S.business_status, website_url=S.website_url, maps_types=S.maps_types
            """
            try:
                client.query(merge_q).result()
            except Exception as e:
                print(f"Merge error: {e}")
    return len(updates)

if __name__ == "__main__":
    enrich_restaurants_by_fhrsid()
