# Plan: Prompt Updates and Agent Enhancement

## Phase 1: Gemini Bulk Analysis Prompt Update
- [ ] Task: Locate the Gemini bulk analysis prompt template (likely in `app/services/bq_utils.py` or `scripts/bq_scripts.py`).
- [ ] Task: Refactor the prompt to explicitly instruct the model to provide the "Final Verdict" at the beginning of the response.
- [ ] Task: Verify the change by running a manual test or existing test that triggers the bulk analysis (if feasible locally) or inspecting the prompt construction in a unit test.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Gemini Bulk Analysis Prompt Update' (Protocol in workflow.md)

## Phase 2: Maps Agent & `get_agent_insight` Update
- [ ] Task: Write failing unit tests for `get_agent_insight` in `app/services/test_agent_orchestrator.py` (or relevant test file) that assert `AddressLine2` and `LocalAuthorityName` are included in the agent prompt.
- [ ] Task: Update `get_agent_insight` in `app/services/agent_orchestrator.py` to extract `AddressLine2` and `LocalAuthorityName` from the input dictionary, handling missing values gracefully.
- [ ] Task: Update the Maps Agent instruction (system instruction) in `app/maps_agent/agent.py` (or where the agent is initialized) to strictly require JSON output format.
- [ ] Task: Update the parsing logic in `app/services/agent_orchestrator.py` (if it exists) to handle the new JSON output format from the agent.
- [ ] Task: Verify unit tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Maps Agent & `get_agent_insight` Update' (Protocol in workflow.md)
