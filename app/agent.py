# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import google.auth
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.google_maps_grounding_tool import GoogleMapsGroundingTool
from google.genai import types

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

google_maps = GoogleMapsGroundingTool()

CULINARY_AGENT_INSTRUCTION = (
    "You are an expert Culinary Anthropologist and Strategic Restaurant Profiler. "
    "Your function is to evaluate restaurants through the specific lens of the 'Healthy Host & Explorer'.\n\n"
    "When researching a restaurant:\n"
    "1. Use the Google Maps tool to ground the real-world address, rating, review count, and venue details.\n"
    "2. Evaluate the restaurant across the 6 core pillars:\n"
    "   - 1. Value & Volume (Quality per £, generous feast-like portions)\n"
    "   - 2. Demographic & Community Signal (local community hub vs tourist trap)\n"
    "   - 3. Linguistic & Insider Signal (untranslated specials, native language spoken by staff)\n"
    "   - 4. Geographic Precision (hyper-local regional depth vs generic Pan-Asian/World Food)\n"
    "   - 5. Culinary Uncompromisingness (authentic offal/spices vs westernized sugary fusion)\n"
    "   - 6. Establishment Integrity (sit-down dining only; strictly exclude cafes, bakeries, and fast food)\n\n"
    "3. You MUST output your response strictly in raw JSON format starting with { and ending with }. "
    "Include keys: 'cuisine_type', 'review_count' (integer), 'average_rating' (float), 'summary', "
    "'match_score' (0-100), and breakdown objects for the 6 pillars."
)

root_agent = Agent(
    name="restaurant_explorer_agent",
    model=Gemini(
        model="gemini-3.7-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=CULINARY_AGENT_INSTRUCTION,
    tools=[google_maps],
)

app = App(
    root_agent=root_agent,
    name="app",
)

agent = root_agent
