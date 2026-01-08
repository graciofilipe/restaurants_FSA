# Plan: Agent Insights Generation

## Phase 1: Infrastructure & Data Access (BigQuery) [checkpoint: 2fa90b5]
- [x] Task: Create BigQuery table `restaurant_agent_insights` with the specified schema.
- [x] Task: Implement `upsert_agent_insight` function in `app/services/bq_utils.py` to handle data persistence.
- [x] Task: Write unit tests for `upsert_agent_insight` ensuring correct upsert logic and data types.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Infrastructure & Data Access (BigQuery)' (Protocol in workflow.md)

## Phase 2: Agent Logic & Data Processing [checkpoint: 0b13a7b]
- [x] Task: Update or create a service to orchestrate calls to the Maps agent for a list of restaurants.
- [x] Task: Implement parsing logic to extract `cuisine_type`, `review_count`, and `average_rating` from the agent's raw text response.
- [x] Task: Write unit tests for the parsing logic with various agent response scenarios (success, partial data, failure).
- [x] Task: Write unit tests for the agent orchestration service, mocking the Maps agent API.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Agent Logic & Data Processing' (Protocol in workflow.md)

## Phase 3: UI Enhancement (Streamlit) [checkpoint: 91e0126]
- [x] Task: Modify the main results table in `app/ui/st_app.py` to enable row selection via checkboxes.
- [x] Task: Implement the "Generate Agent Insights" button and its associated callback/logic.
- [x] Task: Integrate the progress bar and success/error notifications during the agent processing loop.
- [x] Task: Write integration tests (or UI logic tests) to verify selection state and button behavior.
- [x] Task: Conductor - User Manual Verification 'Phase 3: UI Enhancement (Streamlit)' (Protocol in workflow.md)

## Phase 4: Deployment & Final Verification
- [ ] Task: Verify all tests pass locally.
- [ ] Task: Deploy the updated application to Cloud Run using the existing CI/CD pipeline (`cloudbuild.yaml`).
- [ ] Task: Perform end-to-end verification in the production environment: select restaurants, generate insights, and verify BQ table updates.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Deployment & Final Verification' (Protocol in workflow.md)
