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
  AI.GENERATE( ('''Use Google Search and Google Maps to find information about this London restaurant:\n\nRestaurant Name: ''',businessname,''' \n Location: ''',address, ''' \n\nUse Google searches and Google Maps data to evaluate the restaurant based on these criteria: \n Value for Money: Affordable with generous portions for the price. \n Location: In an area with high restaurant competition and some evidence of local native population for the cuisine the restaurant serves. \n Restaurant Type: should NOT be a fast-food, take-away-only, café, bar, pub, brunch place, coffee shop, or pastry shop. \n Ambiance and Style: It should be a casual place for locals, it should NOT be luxurious, high-end, fancy, or sophisticated. \n Customers: Frequented by local, middle/working-class patrons culturally aligned with the cuisine. \n Perform detailed research first, but your final output string must START with the verdict. \n Format: \n FINAL VERDICT: [One of "REJECTED", "Probably Rejected", "Maybe Accepted", "ACCEPTED!", "UNSURE"] \n [Your concise justification and summary]'''),
    connection_id => '{connection_id}',
    endpoint => 'https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/gemini-2.5-pro-preview',
    model_params => JSON '''{{
                              "systemInstruction": {{
                                "parts": [
                                  {{
                                    "text": "You are a restaurant evaluation agent. Use Google Search and Google Maps combined to evaluate restaurants against user-provided criteria. All information must be exclusively from search results and Maps. Generate precise Google Search queries to evaluate based on the criteria, and use Google Maps to understand ratings and customer reviews. Your evaluation should be methodical. Your response MUST START with the final verdict line."
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
  UPDATE SET T.gemini_insights = S.gemini_insights
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
