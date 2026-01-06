# Implementation Plan - Re-architect for Automated Ingestion and Controlled Review

## Phase 1: Configuration & Backend Setup [checkpoint: ]
- [x] Task: Create BigQuery Configuration Table.
    - [x] Define schema for `config_search_params`.
    - [x] Create `scripts/setup_config_table.py` to create the table and populate it with initial values (using existing hardcoded coords as default).
    - [x] Execute the script to set up the environment.
- [x] Task: Create Automated Fetch Script (`app/cron/fetch_weekly.py`).
    - [x] Create `app/cron/` directory and `__init__.py`.
    - [x] Implement logic to read from `config_search_params`.
    - [x] Reuse/Refactor `fetch_data_for_all_coordinates` and `process_and_update_master_data` to be callable from this script without UI dependencies.
    - [x] Ensure `first_seen` is set correctly.
- [x] Task: Verify Fetch Script.
    - [x] Run `python -m app.cron.fetch_weekly` locally and verify BigQuery updates.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Configuration & Backend Setup' (Protocol in workflow.md) [checkpoint: 1d8c18f]

## Phase 2: Frontend Re-architecture (Streamlit) [checkpoint: ]
- [x] Task: Refactor `st_app.py` Layout.
    - [x] Remove "Fetch Data" inputs and button.
    - [x] Create "Configure Review" section (Sidebar or Top).
    - [x] Implement `st.date_input` for "First Seen After".
    - [x] Implement `st.multiselect` for "Review Status".
    - [x] Implement `st.multiselect` for "Exclude Local Authorities" (Query BQ for distinct values).
- [x] Task: Implement "Load Data" Logic.
    - [x] Update `load_filtered_data_from_bq` (or create new) to accept `excluded_authorities` list.
    - [x] Wire "Load Data" button to fetch and store in `st.session_state['review_data']`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Frontend Re-architecture' (Protocol in workflow.md) [checkpoint: d202d26]

## Phase 3: Integration & Cleanup [checkpoint: ]
- [ ] Task: Wire Actions to Session Data.
    - [ ] Update "Run Gemini Analysis" to process `st.session_state['review_data']`.
    - [ ] Update "Export CSV" to export `st.session_state['review_data']`.
- [ ] Task: Deployment Prep.
    - [ ] Ensure `Dockerfile` supports running the app (unchanged).
    - [ ] Document the command for creating the Cloud Run Job: `gcloud run jobs create ... --command "python -m app.cron.fetch_weekly"`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Cleanup' (Protocol in workflow.md)

## Phase 4: Final Verification [checkpoint: ]
- [ ] Task: Run full test suite.
- [ ] Task: Deployment.
    - [ ] Deploy Streamlit App to Cloud Run Service.
    - [ ] (Optional) Create/Update Cloud Run Job for automation.
- [ ] Task: Verify.
    - [ ] Verify UI flows on deployed app.
    - [ ] Verify Job execution (manual trigger via console/CLI).
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification' (Protocol in workflow.md)
