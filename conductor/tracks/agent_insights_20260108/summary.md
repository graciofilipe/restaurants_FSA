# Track Summary: Agent Insights Generation

**Status:** Completed
**Date:** 2026-01-09

## Overview
This track integrated a remote Vertex AI Agent (built with Google ADK) into the FSA Restaurant Reviewer application. The goal was to provide "deep research" insights for selected restaurants, going beyond standard data by utilizing the agent's Google Maps grounding capabilities.

## Key Achievements
1.  **Backend Integration:** Implemented `get_agent_insight` in `app/services/agent_orchestrator.py` to communicate with the deployed Agent Engine.
2.  **SDK Modernization:** Successfully migrated from the legacy `ReasoningEngine` client to the cloud-native `vertexai.Client` and `AgentEngine` SDK. This was crucial for supporting the asynchronous streaming interface of the ADK agent and resolving production errors.
3.  **Data Persistence:** Created a new BigQuery table `restaurant_agent_insights` and implemented upsert logic to store agent findings (Cuisine, Review Count, Rating) linked to the restaurant's FHRSID.
4.  **UI Implementation:** Added a dedicated "Agent Research" tab in the Streamlit application, allowing users to trigger analysis for selected restaurants or in batch mode.
5.  **Robustness:** Implemented robust response parsing to handle JSON output from the agent, including handling of markdown code blocks and stream chunks.

## Artifacts
- **Code:** `app/services/agent_orchestrator.py`, `app/services/bq_utils.py` (updated), `app/ui/st_app.py` (updated).
- **Tests:** `app/services/test_agent_orchestrator.py`.
- **Infrastructure:** BigQuery table `filipegracio_fsa_restaurants.restaurant_agent_insights`.
- **Documentation:** Updated `conductor/tech-stack.md` and `GEMINI.md`.

## Lessons Learned
- **SDK Versioning:** The `vertexai` Python SDK is evolving. ADK-based agents require the newer `vertexai.Client` interface (`AgentEngine`) rather than the older `ReasoningEngine` class, especially for async operations.
- **Stream Parsing:** Agent responses can come in various chunk formats (dictionaries, objects). Robust parsing logic is essential to handle different response shapes from the underlying API.
