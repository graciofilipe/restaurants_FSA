# SQL Scripts for Gemini Enrichment Process

# SCRIPT 1: Identify Recent Restaurants
# Parameters: project_id, dataset_id, source_table, target_table_recents, filter_condition
SCRIPT_IDENTIFY_RECENTS = """
CREATE OR REPLACE TABLE
`{project_id}.{dataset_id}.{target_table_recents}` AS
SELECT
  fhrsid,
  businessname,
  CONCAT( COALESCE(addressline1, ', '), ' ', COALESCE(addressline2, ', '), ' ', COALESCE(addressline3, ', '), ' ', COALESCE(postcode, ', '), ' ', COALESCE(localauthorityname, ' ')) AS address,
  manual_review
FROM
  `{project_id}.{dataset_id}.{source_table}`
WHERE
  {filter_condition}
"""

# SCRIPT 2: Generate Gemini Insights
# Parameters: project_id, dataset_id, source_table_recents, target_table_insights, connection_id, model_endpoint
SCRIPT_GENERATE_INSIGHTS = """
CREATE OR REPLACE TABLE
`{project_id}.{dataset_id}.{target_table_insights}` AS
SELECT
  fhrsid, 
  AI.GENERATE( ('''
  CONTEXT: 
  You are evaluating this restaurant for "The Healthy Host & Explorer" user persona.
  
  Restaurant Details:
  Name: ''',businessname,'''
  Address: ''',address,'''
  
  THE USER PROFILE (The Lens):
  1. Value & Generosity: Seeks feast-like portions, high "Quality per Pound". Avoids "precious" tiny plates.
  2. Native Enclave: Seeks community institutions full of locals/families. Avoids influencer vibes.
  3. Uncompromising Specificity: Rejects generic labels. Seeks distinct regional origin (e.g. "Sicilian", not "Italian").
  4. Establishment Integrity: Must be a full-service sit-down restaurant. No cafes, delis, or fast food.

  EVALUATION PILLARS (The 6 Metrics):
  1. VALUE & PORTION (Quality per £): Bounty vs Transaction.
  2. DEMOGRAPHIC SIGNAL: Native families/Elders vs Tourists/Influencers.
  3. LINGUISTIC SIGNAL: Untranslated menu/Native speech vs English only.
  4. GEOGRAPHIC PRECISION: Specific region (e.g. "Chengdu") vs Generic country.
  5. CULINARY UNCOMPROMISINGNESS: Offal/Spiciness/Funk vs Westernized/Sweet.
  6. ESTABLISHMENT INTEGRITY: Sit-down/Table Service vs Carry-out/Cafe.

  OUTPUT FORMAT RULES:
  - You must strictly output ONLY a valid JSON object.
  - Do NOT include markdown formatting (```json) or introductory text.
  - Use the exact keys defined below.

  JSON Schema:
  {{
      "match_score": Integer (0-100, holistic score based on all pillars),
      "1_value_and_volume": {{
          "rating": Integer (1-5, 1=Stingy, 5=Generous Feast),
          "verdict": String
      }},
      "2_demographic_community": {{
          "score": Integer (1-5, 1=Tourists, 5=Native Community Hub),
          "evidence": String
      }},
      "3_linguistic_signal": {{
          "score": Integer (1-5, 1=English Only, 5=Untranslated/Native Dominant),
          "menu_type": String
      }},
      "4_geographic_precision": {{
          "region_identified": String,
          "specificity_level": String (Enum: ["GENERIC_NATIONAL", "BROAD_REGIONAL", "HYPER_LOCAL_CITY"])
      }},
      "5_culinary_uncompromisingness": {{
          "score": Integer (1-5, 1=Westernized/Safe, 5=Challenging/Authentic),
          "pander_check": String
      }},
      "6_establishment_integrity": {{
          "is_sit_down_restaurant": Boolean,
          "type": String (Enum: ["RESTAURANT_DINING", "CAFE_BAKERY_DELI", "FAST_FOOD_JOINT"])
      }},
      "summary_reasoning": String (Concise verdict: Does it balance Insider Fame with Mainstream Obscurity?)
  }}
  '''),
    connection_id => '{connection_id}',
    endpoint => 'https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_endpoint}',
    model_params => JSON '''{{
                              "systemInstruction": {{
                                "parts": [
                                  {{
                                    "text": "You are an expert Culinary Anthropologist and Strategic Restaurant Profiler. Your function is to filter the real world through the specific lens of the ''Healthy Host & Explorer''. Output strictly valid parseable JSON."
                                  }}
                                ]
                              }},
                              "generationConfig": {{
                                "temperature": 0.4,
                                "maxOutputTokens": 65535,
                                "topP": 0.8
                              }},
                              "safetySettings": [
                                {{ "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF" }},
                                {{ "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF" }},
                                {{ "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF" }},
                                {{ "category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF" }}
                              ],
                              "tools": [
                                {{ "googleSearch": {{}} }}
                              ]
                            }}''').result AS gemini_insights
FROM
`{project_id}.{dataset_id}.{source_table_recents}`
"""

# SCRIPT 3: Merge Insights back to Master
# Parameters: project_id, dataset_id, source_table_insights, target_table_master
SCRIPT_MERGE_INSIGHTS = """
MERGE `{project_id}.{dataset_id}.{target_table_master}` T
USING `{project_id}.{dataset_id}.{source_table_insights}` S
ON T.fhrsid = S.fhrsid
WHEN MATCHED THEN
  UPDATE SET T.gemini_insights_structured = S.gemini_insights
"""

# SCRIPT 4: Bulk Update Manual Reviews
# Parameters: project_id, dataset_id, target_table, source_table_temp
SCRIPT_BULK_UPDATE_MERGE = """
MERGE `{project_id}.{dataset_id}.{target_table}` T
USING `{project_id}.{dataset_id}.{source_table_temp}` S
ON T.fhrsid = S.fhrsid
WHEN MATCHED THEN
  UPDATE SET T.manual_review = S.manual_review
"""
