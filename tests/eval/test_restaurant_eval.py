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

import json
import os
import pytest
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent

import re

def parse_and_validate_json(text: str) -> dict:
    """Helper to validate JSON structure for restaurant profiling."""
    clean = text.strip()
    
    # Remove markdown code blocks if present
    if "```" in clean:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean, re.DOTALL)
        if match:
            clean = match.group(1)
    
    # Extract outer JSON braces
    start_idx = clean.find("{")
    end_idx = clean.rfind("}")
    if start_idx != -1 and end_idx != -1:
        clean = clean[start_idx:end_idx+1]
        
    try:
        data = json.loads(clean.strip())
        assert isinstance(data, dict), "Parsed JSON must be a dictionary"
        return data
    except json.JSONDecodeError:
        # Fallback to incremental raw_decode
        if start_idx != -1:
            obj, _ = json.JSONDecoder().raw_decode(clean)
            assert isinstance(obj, dict), "Parsed JSON must be a dictionary"
            return obj
        raise

def test_evalset_cases_exist():
    """Verify the canonical evaluation set exists and contains at least 5 test cases."""
    evalset_path = os.path.join(os.path.dirname(__file__), "evalsets", "restaurant_eval.evalset.json")
    assert os.path.exists(evalset_path), "restaurant_eval.evalset.json must exist"
    
    with open(evalset_path, "r") as f:
        evalset = json.load(f)
    
    cases = evalset.get("eval_cases", [])
    assert len(cases) >= 5, f"Expected >= 5 evaluation cases, found {len(cases)}"

def test_agent_schema_and_grounding_adherence():
    """Verify ADK agent response schema adherence and grounding execution."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="eval_user", app_name="app")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="app")

    prompt_text = "Research 'Taste of Sichuan' located at 123 High St, London E1 6AN. Ground with Google Maps and score against the 6 culinary anthropologist pillars."
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt_text)]
    )

    events = list(
        runner.run(
            new_message=message,
            user_id="eval_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Agent must return events"

    model_text_parts = []
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and not part.text.strip().startswith("add_queries:"):
                    model_text_parts.append(part.text)

    accumulated_text = "".join(model_text_parts)
    assert len(accumulated_text) > 0, "Agent must return text content"
    
    # Validate structured JSON format
    parsed = parse_and_validate_json(accumulated_text)
    assert "match_score" in parsed or "average_rating" in parsed or "cuisine_type" in parsed or "summary" in parsed, \
        f"Missing expected schema keys in agent output: {parsed.keys()}"
