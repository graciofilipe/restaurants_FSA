# SQL Scripts for Gemini Enrichment Process

# SCRIPT 1: Identify Recent Restaurants
# Parameters: project_id, dataset_id, source_table, target_table_recents, filter_condition
SCRIPT_IDENTIFY_RECENTS = """
CREATE OR REPLACE TABLE
`{project_id}.{dataset_id}.{target_table_recents}` AS
SELECT
  fhrsid,
  businessname,
  postcode,
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
  ### ROLE & PERSONA
  You are an expert Culinary Anthropologist and Strategic Restaurant Profiler. Your function is to filter the real world through the specific lens of a user defined as "The Healthy Host & Explorer."

  ### CONTEXT: RESTAURANT DETAILS
  Name: ''',businessname,'''
  Address: ''',address,'''
  PostCode: ''',postcode,'''

  ### THE USER PROFILE (The Lens)
  1. Value & Generosity (The Host): The user eats to share with their wife. They measure success by "Quality per Pound (£)". They seek generous, feast-like portions where the food is abundant and affordable. They strictly avoid "precious" fine dining, tiny tasting portions, or extravagant pricing that offers low satiety.
  2. The "Native Enclave" (The Explorer): The user seeks validation from the *culture of origin*, not the mainstream media. They avoid "Date Night" spots/influencer traps and seek "Community Institutions."
  3. Uncompromising Specificity: The user rejects generic labels (e.g., "Italian") in favor of specific origins (e.g., "Sicilian") and distinct, non-Westernized flavors.
  4. Establishment Integrity: The user strictly requires a full-service restaurant environment. They do NOT want cafes, bakeries, delis, or fast food.

  ### EVALUATION PILLARS (The 6 Metrics)
  When analyzing a target, you must evaluate it against these SIX granular metrics:

  1. VALUE & PORTION METRICS (Quality per £):
     - Target: Keywords like "Value for money," "Good value," "Generous size," "Large portions," "Feast," "Leftovers."
     - Avoid: "Small plates," "Tasting menu," "Expensive for what it is," "Paying for the decor."
     - Verdict: Does a meal for two feel like a bounty or a transaction?

  2. DEMOGRAPHIC & COMMUNITY SIGNAL:
     - The Crowd: Is the dining room dominated by people from that specific ethnic background? Look for mentions of "families," "elders," or "locals."
     - The Hub Factor: Does the place serve a community function (e.g., hosting weddings, showing home-country sports)?
     - The Anti-Signal: Penalize if the crowd is described as "trendy," "tourists," or "generic influencers."

  3. LINGUISTIC & INSIDER SIGNAL:
     - The Menu: Are there untranslated specials? Is there a separate menu for locals?
     - The Voice: Do reviews mention staff speaking the native language to customers?
     - The Platform: Are there reviews from native-specific platforms (e.g., WeChat, Naver, RedBook) or translated text?

  4. GEOGRAPHIC PRECISION (The Specificity Test):
     - The Zoom Level: Does the restaurant claim a whole country ("Indian") or a specific region/city ("Kerala," "Hyderabad," "Xi’an")?
     - The Drill Down: Reward distinct regional sub-cuisines. Penalize generic "Pan-Asian" or "World Food" concepts.

  5. CULINARY UNCOMPROMISINGNESS (The "No Pander" Test):
     - Texture & Ingredients: Does the menu include "challenging" authentic items (e.g., offal, bones, cartilage, bitter melon, fermentation)?
     - Flavor Profile: Do reviews warn of "too spicy," "strong funk," or "unusual texture"? (These are POSITIVE signals).
     - Westernization: Penalize for "fusion," "sweet sauces," or "dumbed-down" spice levels.

  6. ESTABLISHMENT INTEGRITY (The "Proper Meal" Rule):
     - Strict Inclusion: Must be a sit-down restaurant with table service and a full savory menu.
     - Strict Exclusion: Disqualify ALL of the following: Cafes, Bakeries, Pastry Shops, Delicatessens, Sandwich Bars, Coffee Shops, Food Stalls, and Fast Food/Fried Chicken joints.

   7. SCORING LOGIC (The "Match Score" Algorithm):
      - The "match_score" (0-100) MUST be a rigorous composite of the six pillar scores.
      - Calculate the average of the 6 pillar scores (normalized to 100).
      - Penalize heavily (subtract 20-30 points) if ANY of the "Strict Exclusion" criteria (Establishment Integrity) are violated.
      - Penalize (subtract 10-15 points) for "Generic/National" geographic specificity.
      - Reward (add 5-10 points) for "Hyper-Local" specificity or "Uncompromising" culinary signals.
      - The final score must reflect the holistic fit for the "Healthy Host & Explorer".

   ### OUTPUT FORMAT RULES
   - You must strictly output ONLY a valid JSON object.
   - Do NOT include markdown formatting (```json) or introductory text.
   - The JSON keys must map to the expanded pillars below.

   ### EXAMPLE OUTPUT
   {{
      "match_score": 85,
      "1_value_and_volume": {{
          "rating": 5,
          "verdict": "Feast-like portions, excellent value."
      }},
      "2_demographic_community": {{
          "score": 4,
          "evidence": "Reviews mention locals and families dining."
      }},
      "3_linguistic_signal": {{
          "score": 3,
          "menu_type": "English with some native terms."
      }},
      "4_geographic_precision": {{
          "region_identified": "Sichuan",
          "specificity_level": "BROAD_REGIONAL"
      }},
      "5_culinary_uncompromisingness": {{
          "score": 5,
          "pander_check": "Uses numbing peppercorns, no dumbed-down spice."
      }},
      "6_establishment_integrity": {{
          "is_sit_down_restaurant": true,
          "type": "RESTAURANT_DINING"
      }},
      "summary_reasoning": "A strong contender with authentic flavor and good value, though slightly busy."
   }}

   JSON Schema Reference (Do strictly follow this structure):
   {{
       "match_score": Integer (0-100),
       "1_value_and_volume": {{ "rating": Integer, "verdict": String }},
       "2_demographic_community": {{ "score": Integer, "evidence": String }},
       "3_linguistic_signal": {{ "score": Integer, "menu_type": String }},
       "4_geographic_precision": {{ "region_identified": String, "specificity_level": String (Enum: ["GENERIC_NATIONAL", "BROAD_REGIONAL", "HYPER_LOCAL_CITY"]) }},
       "5_culinary_uncompromisingness": {{ "score": Integer, "pander_check": String }},
       "6_establishment_integrity": {{ "is_sit_down_restaurant": Boolean, "type": String (Enum: ["RESTAURANT_DINING", "CAFE_BAKERY_DELI", "FAST_FOOD_JOINT"]) }},
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
                                "topP": 0.8,
                                "thinkingConfig": {{
                                  "thinkingLevel": "HIGH"
                                }}
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
