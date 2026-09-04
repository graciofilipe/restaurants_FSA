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

import inspect
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent as explorer_agent
from app.maps_agent.agent import root_agent as maps_agent
from app.services.bq_utils import execute_gemini_enrichment


def test_explorer_agent_model_is_gemini_3_8_flash():
    """Verify that root_agent in app/agent.py is configured with gemini-3.8-flash."""
    assert explorer_agent.name == "restaurant_explorer_agent"
    assert hasattr(explorer_agent, "model")
    # explorer_agent uses Gemini model wrapper
    if hasattr(explorer_agent.model, "model"):
        assert explorer_agent.model.model == "gemini-3.8-flash"
    else:
        assert explorer_agent.model == "gemini-3.8-flash"


def test_maps_agent_model_is_gemini_3_8_flash():
    """Verify that root_agent in app/maps_agent/agent.py is configured with gemini-3.8-flash."""
    assert maps_agent.name == "restaurant_maps_agent"
    assert maps_agent.model == "gemini-3.8-flash"


def test_bq_utils_default_model_endpoint():
    """Verify that execute_gemini_enrichment defaults to gemini-3.8-flash."""
    sig = inspect.signature(execute_gemini_enrichment)
    assert sig.parameters["model_endpoint"].default == "gemini-3.8-flash"


@patch("app.services.bq_utils.bigquery.Client")
def test_bq_utils_query_generates_gemini_3_8_flash_endpoint(mock_client_cls):
    """Verify that execute_gemini_enrichment injects gemini-3.8-flash into the BQ query."""
    mock_client = mock_client_cls.return_value
    mock_job = MagicMock()
    mock_client.query.return_value = mock_job

    success = execute_gemini_enrichment(
        project_id="test-proj",
        dataset_id="test-dataset",
        master_table_id="test-master",
    )
    assert success is True
    # Check that the second query (SCRIPT_GENERATE_INSIGHTS) references gemini-3.8-flash
    call_args_list = mock_client.query.call_args_list
    assert len(call_args_list) >= 2
    insights_query = call_args_list[1][0][0]
    assert "publishers/google/models/gemini-3.8-flash" in insights_query


def test_eval_config_judge_model():
    """Verify that eval_config.json specifies gemini-3.8-flash as the judge model."""
    eval_config_path = os.path.join(
        os.path.dirname(__file__), "eval", "eval_config.json"
    )
    with open(eval_config_path, "r") as f:
        config = json.load(f)

    judge_model = (
        config.get("criteria", {})
        .get("rubric_based_final_response_quality_v1", {})
        .get("judgeModelOptions", {})
        .get("judgeModel")
    )
    assert judge_model == "gemini-3.8-flash"


def test_live_gemini_3_8_flash_adk_invocation():
    """Test live ADK execution using gemini-3.8-flash and GoogleMapsGroundingTool."""
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(
        user_id="test_upgrade_user", app_name="app"
    )
    runner = Runner(agent=explorer_agent, session_service=session_service, app_name="app")

    prompt = (
        "Ground Dishoom Covent Garden London and return brief JSON with cuisine_type."
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    events = list(
        runner.run(
            new_message=message,
            user_id="test_upgrade_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    )
    assert len(events) > 0, "Expected non-empty events from runner"

    text_parts = []
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and not part.text.strip().startswith("add_queries:"):
                    text_parts.append(part.text)

    output = "".join(text_parts)
    assert len(output) > 0, "Expected response text from gemini-3.8-flash"
