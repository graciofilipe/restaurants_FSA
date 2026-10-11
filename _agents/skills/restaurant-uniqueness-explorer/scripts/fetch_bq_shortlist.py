#!/usr/bin/env python3
"""Fetches or enriches shortlisted restaurants from BigQuery `fsa_master`.

Supports two modes:
1. Lookup specific restaurant names passed by the user (`--names "Name 1" "Name 2"`).
2. Pull top shortlisted candidates from `fsa_master` filtered by outcode, borough,
   keyword search, minimum match_score, or top predicted_user_rating.
"""

import argparse
import json
import subprocess
import sys
from google.api_core import exceptions as gcp_exceptions
from google.cloud import bigquery
from google.oauth2 import credentials as oauth2_credentials

DEFAULT_TABLE = "filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master"


def build_query(args: argparse.Namespace) -> tuple[str, list]:
    table = args.table
    conditions = ["in_scope IS NOT FALSE"]
    params = []

    if args.names:
        name_clauses = []
        for idx, name in enumerate(args.names):
            param_name = f"name_{idx}"
            name_clauses.append(f"LOWER(businessname) LIKE LOWER(@{param_name})")
            params.append(bigquery.ScalarQueryParameter(param_name, "STRING", f"%{name.strip()}%"))
        conditions.append("(" + " OR ".join(name_clauses) + ")")

    if args.outcode:
        conditions.append("UPPER(TRIM(postcode)) LIKE UPPER(@outcode_prefix)")
        params.append(
            bigquery.ScalarQueryParameter("outcode_prefix", "STRING", f"{args.outcode.strip()}%")
        )

    if args.borough:
        conditions.append("LOWER(localauthorityname) LIKE LOWER(@borough)")
        params.append(
            bigquery.ScalarQueryParameter("borough", "STRING", f"%{args.borough.strip()}%")
        )

    if args.search:
        conditions.append(
            "(LOWER(businessname) LIKE LOWER(@search) "
            "OR LOWER(COALESCE(pillar_geo_region, '')) LIKE LOWER(@search) "
            "OR LOWER(COALESCE(summary_reasoning, '')) LIKE LOWER(@search))"
        )
        params.append(
            bigquery.ScalarQueryParameter("search", "STRING", f"%{args.search.strip()}%")
        )

    if args.min_match_score is not None:
        conditions.append("COALESCE(match_score, 0) >= @min_match_score")
        params.append(
            bigquery.ScalarQueryParameter("min_match_score", "INT64", args.min_match_score)
        )

    if args.unrated_only:
        conditions.append("(user_rating IS NULL OR TRIM(CAST(user_rating AS STRING)) = '')")

    where_sql = " AND ".join(conditions)
    query = f"""
    SELECT
      fhrsid,
      businessname,
      addressline1,
      addressline2,
      addressline3,
      postcode,
      localauthorityname,
      maps_rating,
      maps_reviews,
      price_level,
      maps_url,
      website_url,
      user_rating,
      ROUND(predicted_user_rating, 2) AS predicted_user_rating,
      match_score,
      pillar_geo_region,
      pillar_geo_specificity,
      pillar_value_rating,
      pillar_community_score,
      pillar_linguistic_score,
      pillar_culinary_score,
      summary_reasoning
    FROM `{table}`
    WHERE {where_sql}
    ORDER BY
      COALESCE(predicted_user_rating, 0) DESC,
      COALESCE(match_score, 0) DESC,
      COALESCE(maps_rating, 0) DESC
    LIMIT @limit
    """
    params.append(bigquery.ScalarQueryParameter("limit", "INT64", args.limit))
    return query, params


def _gcloud_token_client(project_id: str) -> bigquery.Client:
    token = subprocess.check_output(
        ["gcloud", "auth", "print-access-token", "filipegracio@google.com"],
        text=True,
    ).strip()
    creds = oauth2_credentials.Credentials(token=token)
    return bigquery.Client(project=project_id, credentials=creds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch or enrich shortlisted restaurants from BigQuery fsa_master."
    )
    parser.add_argument("--table", default=DEFAULT_TABLE, help="BigQuery table path")
    parser.add_argument("--names", nargs="+", help="Specific restaurant names to look up")
    parser.add_argument("--outcode", help="Filter by UK postcode prefix (e.g. SW16, SE1)")
    parser.add_argument("--borough", help="Filter by local authority / borough name")
    parser.add_argument("--search", help="Keyword filter across name, region, or summary")
    parser.add_argument("--min-match-score", type=int, default=None, help="Minimum Gemini match_score")
    parser.add_argument(
        "--unrated-only",
        action="store_true",
        default=False,
        help="Only return restaurants without a personal user_rating",
    )
    parser.add_argument("--limit", type=int, default=5, help="Max number of candidates to return")
    args = parser.parse_args()

    project_id = args.table.split(".")[0]
    query, params = build_query(args)
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    try:
        client = bigquery.Client(project=project_id)
        rows = [dict(r) for r in client.query(query, job_config=job_config).result()]
    except gcp_exceptions.Forbidden:
        client = _gcloud_token_client(project_id)
        rows = [dict(r) for r in client.query(query, job_config=job_config).result()]

    print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
