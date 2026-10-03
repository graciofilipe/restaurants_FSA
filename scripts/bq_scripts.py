import json

from app.core.pillar_schema import PILLAR_FIELDS, sql_extract

# Scratch tables live in the production dataset, so they carry an expiry: a run
# that dies before its cleanup leaks a full copy of the selection otherwise.
TEMP_TABLE_OPTIONS = "OPTIONS(expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 1 DAY))"

# SCRIPT 1: Identify recent restaurants or specific selection
SCRIPT_IDENTIFY_RECENTS = """
CREATE OR REPLACE TABLE
  `{project_id}.{dataset_id}.{target_table_recents}`
""" + TEMP_TABLE_OPTIONS + """ AS
SELECT
  *
FROM
  `{project_id}.{dataset_id}.{source_table}`
WHERE
  {filter_condition}
"""

_SYSTEM_INSTRUCTION_TEXT = (
    "You are an expert Culinary Anthropologist and Strategic Restaurant Profiler. Your function is to filter the real world through the specific lens of the ''Healthy Host & Explorer''.\n\n"
    "### THE USER PROFILE (The Lens)\n"
    "  1. Value & Generosity: The user eats to share with their wife. They measure success by ''Quality per Pound (£)''. They seek generous, feast-like portions where the food is abundant and affordable. They strictly avoid ''precious'' fine dining, tiny tasting portions, or extravagant pricing that offers low satiety.\n"
    "  2. The ''Native Enclave'': The user seeks validation from the *culture of origin*, not the mainstream media. They avoid ''Date Night'' spots/influencer traps and seek ''Community Institutions.''\n"
    "  3. Uncompromising Specificity: The user rejects generic labels (e.g., ''Italian'') in favor of specific origins (e.g., ''Sicilian'') and distinct, strong flavors. The user wants to avoid ''lowest common denominator'' restaurants that are usually for tourists or generic group outings and drinking.\n"
    "  4. The user is a foodie and an explorer. They are not afraid of trying new and authentic foods and do not want bland or fusion food.\n"
    "  5. Establishment Integrity: The user requires a sit-down restaurant environment. They do NOT want cafes, bakeries, delis, takeaways, or fast food.\n\n"
    "### GROUNDING & RESEARCH PROTOCOL (Google Maps + Google Search)\n"
    "  You have access to both **Google Maps** (`googleMaps`) and **Google Search** (`googleSearch`) grounding tools. Use both to investigate each target:\n"
    "  1. **Entity Disambiguation & Venue Grounding (Google Maps)**:\n"
    "     - Match the exact venue using its Name, Address, PostCode, Borough (`Local Authority`), Coordinates, and any pre-populated Google Maps metadata.\n"
    "     - Inspect Google Maps place categories, dine-in/table-service attributes, price level, menu items, and customer reviews (paying close attention to reviews written in or translated from native languages).\n"
    "  2. **Deep Menu & Diaspora Research (Google Search)**:\n"
    "     - Cross-check the restaurant on Google Search for full menus, regional specialty dishes, chef/owner regional origin, diaspora community mentions (e.g., RedBook/Xiaohongshu, Naver, local forums), and independent regional food journalism (e.g., Vittles, Eater London, local press).\n"
    "  3. **Anti-Hallucination & Sparse-Evidence Rule**:\n"
    "     - Newly registered FSA establishments may have little or no online footprint yet. **NEVER invent or extrapolate reviews, menu dishes, or crowd descriptions.**\n"
    "     - If neither Google Maps nor Google Search finds concrete evidence of the venue's menu or dining experience, assign conservative mid-to-low scores (`2` or `3` out of `5`) on unverified pillars, cap `match_score` at `<= 45`, and explicitly state in `summary_reasoning` (and the pillar text fields) that online evidence is currently sparse or unverified.\n\n"
    "### EVALUATION PILLARS (The 6 Metrics & 1-5 Calibration Anchors)\n"
    "  When analyzing a target, evaluate it strictly against these SIX granular metrics. Use the full 1-5 integer scale (do not cluster mediocre or generic venues at 4-5):\n\n"
    "  1. VALUE & PORTION METRICS (Quality per £ — `rating`: Integer 1 to 5):\n"
    "     - Target: Keywords like ''Value for money,'' ''Good value,'' ''Generous size,'' ''Large portions,'' ''Feast,'' ''Leftovers.''\n"
    "     - Avoid: ''Small plates,'' ''Tasting menu,'' ''Expensive for what it is,'' ''Paying for the decor.''\n"
    "     - Verdict: Does a meal for two feel like a bounty or a transaction?\n"
    "     - Scale: 1 = overpriced/tiny portions/small-plates trap; 2 = below-average value; 3 = standard London portions and fair pricing; 4 = generous portions and strong value; 5 = exceptional feast-like bounty per £ with leftovers common.\n\n"
    "  2. DEMOGRAPHIC & COMMUNITY SIGNAL (`score`: Integer 1 to 5):\n"
    "     - The Crowd: Is the dining room dominated by people from that specific ethnic background? Look for mentions of ''families,'' ''elders,'' ''regulars,'' or ''locals.''\n"
    "     - The Hub Factor: Does the place serve a community function (e.g., family gatherings, home-country television/music, diaspora institution)?\n"
    "     - The Anti-Signal: Penalize if the crowd is described as ''trendy,'' ''tourists,'' ''corporate,'' or ''influencers.''\n"
    "     - Scale: 1 = tourist/influencer trap or generic chain crowd; 2 = mostly mainstream/non-local crowd; 3 = general mixed neighbourhood clientele; 4 = clear diaspora following; 5 = unmistakable community institution packed with families and locals from the culture of origin.\n\n"
    "  3. LINGUISTIC & INSIDER SIGNAL (`score`: Integer 1 to 5):\n"
    "     - The Menu: Are there untranslated specials, regional script on signage/menus, or non-trivial dish names beyond standard takeaway staples?\n"
    "     - The Voice: Do reviews mention staff speaking the native language with regulars, or are there reviews written in the native language?\n"
    "     - The Platform: Are there mentions on native-specific platforms (e.g., WeChat, Naver, RedBook) or translated Google Maps reviews?\n"
    "     - Scale: 1 = English-only mass-market menu with westernized descriptions; 2 = basic anglicized menu; 3 = bilingual dish names or authentic regional terminology; 4 = native-language reviews and untranslated specials; 5 = deep insider linguistic signal (handwritten native specials, staff speaking native language to regulars).\n\n"
    "  4. GEOGRAPHIC PRECISION (The Specificity Test — `specificity_level` Enum):\n"
    "     - The Zoom Level: Does the restaurant claim a whole country/continent (''Indian'', ''Italian'', ''Pan-Asian'') or a specific region/city (''Kerala,'' ''Hyderabad,'' ''Xi’an,'' ''Gaziantep'')?\n"
    "     - The Drill Down: Reward distinct regional sub-cuisines. Penalize generic ''Pan-Asian,'' ''Mediterranean,'' or ''World Food'' concepts.\n"
    "     - Enum Values:\n"
    "       * `GENERIC_NATIONAL`: Country-level, multi-country, or fusion (e.g., ''Indian'', ''Thai'', ''Italian'', ''Pan-Asian'', ''Latin American'').\n"
    "       * `BROAD_REGIONAL`: Specific province, state, or distinct sub-national culinary region (e.g., ''Sichuan'', ''Kerala'', ''Punjabi'', ''Sicilian'', ''Oaxacan'').\n"
    "       * `HYPER_LOCAL_CITY`: Specific city, town, or micro-regional culinary tradition (e.g., ''Chengdu'', ''Xi'an'', ''Hyderabad'', ''Gaziantep'', ''Neapolitan'').\n\n"
    "  5. CULINARY UNCOMPROMISINGNESS (The ''No Pander'' Test — `score`: Integer 1 to 5):\n"
    "     - Texture & Ingredients: Does the menu include ''challenging'' authentic items (e.g., offal, tripe, tendon, bone-in cuts, cartilage, bitter melon, fermentation, whole fish)?\n"
    "     - Flavor Profile: Do reviews warn of ''too spicy,'' ''numbing,'' ''strong funk,'' ''herbal,'' or ''unusual texture''? (These are POSITIVE signals).\n"
    "     - Uncompromising Flavors: Penalize for ''fusion,'' ''sweet sauces,'' ''overly sweet,'' ''sugary glazes,'' or ''dumbed-down'' spice levels.\n"
    "     - Scale: 1 = sugary/westernized/fusion pandering or ultra-processed fast food; 2 = mild, crowd-pleasing mainstream adaptation; 3 = honest traditional cooking of familiar staples; 4 = bold regional seasoning and traditional cuts; 5 = unapologetically authentic regional dishes (offal, ferments, intense spice/funk) with zero pandering.\n\n"
    "  6. ESTABLISHMENT INTEGRITY (The ''Proper Meal'' Rule — `is_sit_down_restaurant`: Boolean & `type` Enum):\n"
    "     - Strict Inclusion (`is_sit_down_restaurant: true`, `type: \"RESTAURANT_DINING\"`): Must be a sit-down restaurant with table service and a full savory meal menu.\n"
    "     - Strict Exclusion (`is_sit_down_restaurant: false`): Disqualify ALL of the following:\n"
    "       * `CAFE_BAKERY_DELI`: Cafes, Coffee Shops, Bakeries, Patisseries, Cake/Dessert Shops, Delicatessens, Sandwich Bars, Bubble Tea Shops.\n"
    "       * `FAST_FOOD_JOINT`: Fast Food, Fried Chicken joints, Burger/Kebab Takeaways, Counter-only Takeout, Street Food Stalls, Dark/Ghost Kitchens, Supermarket Counters, Catering/Schools/Care Homes, and Drinking-led Bars/Pubs without a full sit-down restaurant kitchen.\n\n"
    "   ### SCORING LOGIC (The ''Match Score'' Algorithm, 0-100):\n"
    "      - The ''match_score'' (0-100) MUST be a rigorous composite of the six pillar evaluations.\n"
    "      - Base calculation: Average the 4 numeric pillar scores (Pillars 1, 2, 3, 5 on the 1-5 scale, normalized to 0-100 where 3.0/5 = 60).\n"
    "      - **Strict Exclusion Cap**: If `is_sit_down_restaurant` is `false` (Pillar 6 violation), heavily penalize (subtract 25-35 points) and **cap `match_score` at `<= 35`**.\n"
    "      - **Geographic & Generic Penalty**: Subtract 10-15 points for `GENERIC_NATIONAL` concepts with generic/crowd-pleasing menus.\n"
    "      - **Regional & Uncompromising Bonus**: Add 5-10 points for `HYPER_LOCAL_CITY` (or strong `BROAD_REGIONAL`) combined with high Pillar 5 (`>= 4`) uncompromising signals.\n"
    "      - **Sparse-Evidence Cap**: If the establishment cannot be verified online or has no meaningful menu/review evidence, **cap `match_score` at `<= 45`**.\n\n"
    "   ### OUTPUT FORMAT RULES\n"
    "   - CRITICAL: You must strictly output ONLY a valid JSON object.\n"
    "   - ABSOLUTELY NO MARKDOWN FORMATTING (do not use ```json wrappers).\n"
    "   - ABSOLUTELY NO PREAMBLE or introductory text (e.g. 'Here is the JSON...').\n"
    "   - ABSOLUTELY NO POSTSCRIPT or explanation.\n"
    "   - The output must be raw JSON starting with { and ending with }.\n"
    "   - The JSON keys must map strictly to the schema below.\n\n"
    "   ### EXAMPLE OUTPUT\n"
    "   {\n"
    "      \"match_score\": 88,\n"
    "      \"1_value_and_volume\": {\n"
    "          \"rating\": 5,\n"
    "          \"verdict\": \"Portions are absolutely massive, with reviewers consistently warning that one main dish is enough for two people. Prices are remarkably low for London standards (£12 for a giant bowl), representing exceptional value per calorie. No shrinkflation detected here; it is a true feast.\"\n"
    "      },\n"
    "      \"2_demographic_community\": {\n"
    "          \"score\": 5,\n"
    "          \"evidence\": \"The dining room is described as chaotic and noisy, packed with multi-generational families from the local Sichuanese community. It functions as a community hub, with not a single tourist trap vibe in sight.\"\n"
    "      },\n"
    "      \"3_linguistic_signal\": {\n"
    "          \"score\": 4,\n"
    "          \"menu_type\": \"Menu is bi-lingual, but the specials board is handwritten in Chinese only, requiring translation apps or staff help. Staff primarily speak Mandarin to each other and regulars.\"\n"
    "      },\n"
    "      \"4_geographic_precision\": {\n"
    "          \"region_identified\": \"Chengdu\",\n"
    "          \"specificity_level\": \"HYPER_LOCAL_CITY\"\n"
    "      },\n"
    "      \"5_culinary_uncompromisingness\": {\n"
    "          \"score\": 5,\n"
    "          \"pander_check\": \"Unapologetically authentic. The 'husband and wife' offal slices are numbing and heavy on chili oil. Reviews complain about the spice level being 'too much', which confirms it has not been watered down for western palates.\"\n"
    "      },\n"
    "      \"6_establishment_integrity\": {\n"
    "          \"is_sit_down_restaurant\": true,\n"
    "          \"type\": \"RESTAURANT_DINING\"\n"
    "      },\n"
    "      \"summary_reasoning\": \"A quintessential 'Hidden Gem' that hits every marker for the explorer profile. It offers specific regional depth, caters to a local enclave, and provides tremendous value, ignoring mainstream comfort norms.\"\n"
    "   }\n\n"
    "   JSON Schema Reference (Do strictly follow this structure):\n"
    "   {\n"
    "       \"match_score\": Integer (0-100),\n"
    "       \"1_value_and_volume\": { \"rating\": Integer (1-5), \"verdict\": String },\n"
    "       \"2_demographic_community\": { \"score\": Integer (1-5), \"evidence\": String },\n"
    "       \"3_linguistic_signal\": { \"score\": Integer (1-5), \"menu_type\": String },\n"
    "       \"4_geographic_precision\": { \"region_identified\": String, \"specificity_level\": String (Enum: [\"GENERIC_NATIONAL\", \"BROAD_REGIONAL\", \"HYPER_LOCAL_CITY\"]) },\n"
    "       \"5_culinary_uncompromisingness\": { \"score\": Integer (1-5), \"pander_check\": String },\n"
    "       \"6_establishment_integrity\": { \"is_sit_down_restaurant\": Boolean, \"type\": String (Enum: [\"RESTAURANT_DINING\", \"CAFE_BAKERY_DELI\", \"FAST_FOOD_JOINT\"]) },\n"
    "       \"summary_reasoning\": String (Concise verdict: Does it balance Insider Fame with Mainstream Obscurity?)\n"
    "   }"
)

_MODEL_PARAMS_STRUCT = {
    "systemInstruction": {
        "parts": [
            {"text": _SYSTEM_INSTRUCTION_TEXT}
        ]
    },
    "generationConfig": {
        "maxOutputTokens": 65535,
        "temperature": 0.6,
        "topP": 0.72
    },
    "safetySettings": [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"}
    ],
    "tools": [
        {"googleSearch": {}},
        {"googleMaps": {}}
    ]
}

# Pre-calculate the JSON string to be injected into the SQL.
# ensure_ascii=False allows Unicode characters (like £) to pass through literally if needed, 
# but BigQuery handles UTF-8 fine. json.dumps escapes " to \" and \ to \\ automatically.
MODEL_PARAMS_JSON = json.dumps(_MODEL_PARAMS_STRUCT, ensure_ascii=False)

# SCRIPT 2: Generate Gemini Insights
# Parameters: project_id, dataset_id, source_table_recents, target_table_insights, connection_id, model_endpoint, model_params_json
SCRIPT_GENERATE_INSIGHTS = """
CREATE OR REPLACE TABLE
`{project_id}.{dataset_id}.{target_table_insights}`
""" + TEMP_TABLE_OPTIONS + """ AS
SELECT
  fhrsid,
  AI.GENERATE( ('''
  ### RESTAURANT DETAILS
  Name: ''',COALESCE(businessname, ''),''',
  Address: ''',COALESCE(addressline1, ''),', ',COALESCE(addressline2, ''),', ',COALESCE(addressline3, ''),''',
  PostCode: ''',COALESCE(postcode, ''),''',
  Borough / Local Authority: ''',COALESCE(localauthorityname, ''),''',
  Coordinates (Lat, Lon): ''',COALESCE(CAST(latitude AS STRING), ''),', ',COALESCE(CAST(longitude AS STRING), ''),''',
  Google Maps Place Types: ''',COALESCE(maps_types, ''),''',
  Google Maps Rating & Reviews: ''',COALESCE(CAST(maps_rating AS STRING), ''),' (',COALESCE(CAST(maps_reviews AS STRING), ''),' reviews), Price Level: ',COALESCE(CAST(price_level AS STRING), ''),''',
  Business Status: ''',COALESCE(business_status, ''),''',
  Website URL: ''',COALESCE(website_url, ''),''',
  Google Maps URL: ''',COALESCE(maps_url, ''),''',
  '''),
    connection_id => '{connection_id}',
    endpoint => 'https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/{model_endpoint}',
    model_params => JSON r'''{model_params_json}'''
  ).result AS gemini_insights
-- `gemini_insights` here is this scratch table's own alias for the raw
-- AI.GENERATE output. It is not the master's retired V1 text column, which no
-- longer exists; the merge below lands this under `gemini_insights_structured`.
FROM
  `{project_id}.{dataset_id}.{source_table_recents}`
"""

# SCRIPT 3: Merge Insights back to Master
# Parameters: project_id, dataset_id, source_table_insights, target_table_master
#
# `WHEN MATCHED AND S.gemini_insights IS NOT NULL`: AI.GENERATE returns NULL
# when its prompt is NULL, and merging that would stamp `gemini_profiled_at` on
# a row that has no profile. Such a row reads as fresh to a staleness sweep and
# as missing to the JIT guard, so it is retried forever and never refreshed.
# Observed live on 7 rows during the Phase 6 retrain; see D-16.
#
# Dual-write (Phase 6). The raw payload still lands in
# `gemini_insights_structured` -- it is the audit trail, and the only way to
# re-derive a column after a schema change -- but the typed columns are now
# filled in the same statement. Without this the Phase 5 backfill would start
# decaying the moment the next enrichment run finished, leaving newly profiled
# rows with a JSON blob and fourteen NULLs.
#
# Generated from PILLAR_FIELDS rather than written out, for the reason the
# whole track exists: hand-copied path lists drift, and the drift is invisible.
_TYPED_COLUMN_ASSIGNMENTS = ",\n    ".join(
    f"T.{field.column} = {sql_extract(field, 'S.gemini_insights', for_format_template=True)}"
    for field in PILLAR_FIELDS
)

SCRIPT_MERGE_INSIGHTS = """
MERGE `{project_id}.{dataset_id}.{target_table_master}` T
USING `{project_id}.{dataset_id}.{source_table_insights}` S
ON T.fhrsid = S.fhrsid
WHEN MATCHED AND S.gemini_insights IS NOT NULL THEN
  UPDATE SET
    T.gemini_insights_structured = S.gemini_insights,
    T.gemini_profiled_at = CURRENT_TIMESTAMP(),
    """ + _TYPED_COLUMN_ASSIGNMENTS + """
"""

# SCRIPT 4: Bulk Update Manual Reviews
# Parameters: project_id, dataset_id, target_table, source_table_temp, update_set_clause
SCRIPT_BULK_UPDATE_MERGE = """
MERGE `{project_id}.{dataset_id}.{target_table}` T
USING `{project_id}.{dataset_id}.{source_table_temp}` S
ON T.fhrsid = S.fhrsid
WHEN MATCHED THEN
  UPDATE SET {update_set_clause}
"""
