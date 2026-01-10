# Plan: Display Agent Insights on UI

## Phase 1: Backend Integration (BigQuery Data Retrieval) [checkpoint: a49af84]
- [x] Task: Write failing unit tests for `load_specific_agent_insights` in `app/services/test_bq_utils_agent.py`. 5738261
- [x] Task: Implement `load_specific_agent_insights(project_id, dataset_id, fhrsids)` in `app/services/bq_utils.py`. bfd4e86
- [x] Task: Verify unit tests pass and code coverage for the new function is >80%. bfd4e86
- [x] Task: Conductor - User Manual Verification 'Phase 1: Backend Integration (BigQuery Data Retrieval)' (Protocol in workflow.md)

## Phase 2: UI Implementation (Streamlit Results View) [checkpoint: 8234514]
- [x] Task: Write failing tests for UI state management (verifying `session_state.latest_insights` is populated correctly). 52da82e
- [x] Task: Update the "Generate Agent Insights" loop in `app/ui/st_app.py` to collect processed `fhrsid`s and trigger a data fetch upon completion. 52da82e
- [x] Task: Implement the "Latest Batch Insights" expander and table display in `app/ui/st_app.py` below the generation section. 52da82e
- [x] Task: Verify UI responsiveness and correct data filtering (only "just now" batch is shown). 52da82e
- [x] Task: Conductor - User Manual Verification 'Phase 2: UI Implementation (Streamlit Results View)' (Protocol in workflow.md)

## Phase 3: Deployment & Production Verification
- [x] Task: Deploy the updated application to Google Cloud Run using the existing CI/CD pipeline (`cloudbuild.yaml`).
- [ ] Task: Perform final acceptance testing on the production URL: https://restaurants-fsa-tqzsejpoja-nw.a.run.app.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Deployment & Production Verification' (Protocol in workflow.md)
