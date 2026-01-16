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
  AI.GENERATE( ('''Use Google Search and Google Maps to find information about this London restaurant:\n\nRestaurant Name: ''',businessname,''' \n Location: ''',address, ''' \n\nResearch and evaluate this restaurant based on the following criteria:\n1. Affordability & Portions: Is it good value? Big portions?\n2. Authenticity: Is it "Westernized" or does it serve authentic regional dishes? Does it attract a native local community?\n3. Atmosphere: Is it a casual, "hole-in-the-wall" or "no-frills" spot? Or is it fancy/upscale?\n4. Vibe: Is it a sit-down place? (Not just takeaway/cafe)\n\nOUTPUT FORMAT: \nYou must strictly output ONLY a valid JSON object. Do not include markdown formatting (```json).\n\nSchema:\n{{\n    "match_score": Integer (0-100, holistic score based on strict adherence to ALL user preferences: Affordability + Portions, Specific Cultural Authenticity, Sit-down Restaurant Type, and Casual Vibe),\n    "cultural_authenticity_rating": Integer (1-5, 5=Evidence of native patronage/languages), \n    "establishment_type": String (Enum: ["RESTAURANT_SITDOWN", "TAKEAWAY_ONLY", "CAFE_BRUNCH", "PUB_BAR", "FAST_FOOD", "OTHER"]),\n    "atmosphere": String (Enum: ["DIVE_HOLE_IN_WALL", "CASUAL_NO_FRILLS", "BUSTLING_FAMILY", "UPSCALE_FANCY", "GENERIC_MODERN", "TOURIST_TRAP"]),\n    "portion_size": String (Enum: ["SMALL_PLATES", "AVERAGE", "GENEROUS", "UNKNOWN"]),\n    "value_rating": Integer (1-5, 5=Cheap/Generous, 1=Expensive), \n    "summary_reasoning": String (Concise verdict)\n}}'''),
    connection_id => '{connection_id}',
    endpoint => 'https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_endpoint}',
    model_params => JSON '''{{
                              "systemInstruction": {{
                                "parts": [
                                  {{
                                    "text": "You are a specialized restaurant profiler. Your result must be a valid, parseable JSON object. Do not include any introductory text or markdown code blocks. Focus heavily on detecting ''Authenticity'' and ''Specific Emigrant Communities''."
                                  }}
                                ]
                              }},
                              "generationConfig": {{
                                "temperature": 0.6,
                                "maxOutputTokens": 65535,
                                "topP": 0.72
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
