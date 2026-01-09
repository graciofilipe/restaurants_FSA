# Implementation Plan - Multi-Coordinate Search Support

## Phase 1: Database & Configuration [checkpoint: 3bfa77a]
- [x] Task: Update `config_search_params` Table Schema.
    - [x] Create `scripts/update_bq_schema.py` (or modify setup) to recreate/migrate the table.
    - [x] Schema change: Ensure `coordinates` is handled as individual rows or the table structure supports multiple search targets.
    - [x] Test: Verify the table can store multiple coordinate rows.
- [x] Task: Update Data Ingestion Script.
    - [x] Modify `scripts/setup_config_table.py` to accept a list of coordinate pairs.
    - [x] Implement logic to iterate and insert/load these pairs into BigQuery.
    - [x] Test: Run the script with a sample list (mocked or dry-run) and verify data structure.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Database & Configuration' (Protocol in workflow.md) [checkpoint: 3bfa77a]

## Phase 2: Backend Logic Update [checkpoint: 3466dbc]
- [x] Task: Refactor `fetch_weekly.py` for Multi-Row Processing.
    - [x] Create/Update `app/cron/test_fetch_weekly.py` to mock `fetch_config_params` returning multiple rows.
    - [x] Implement loop in `app/cron/fetch_weekly.py` to iterate through all config rows.
    - [x] Implement "Continue on Error" logic (try/except block inside the loop).
    - [x] Ensure `process_and_update_master_data` is called for each valid coordinate pair.
    - [x] Test: Verify logic handles multiple rows and continues after a simulated error in one.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Backend Logic Update' (Protocol in workflow.md) [checkpoint: 3466dbc]

## Phase 3: Deployment & Verification [checkpoint: ]
- [x] Task: Deployment Prep.
    - [x] Verify `Dockerfile` and `cloudbuild.yaml` (should be unchanged, but good to check).
- [x] Task: Final Verification.
    - [x] Run full test suite.
    - [x] Deploy changes.
    - [x] Trigger a manual run of the Cloud Run Job (or script) to verify end-to-end multi-coordinate fetching.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Deployment & Verification' (Protocol in workflow.md)
    - [ ] Run full test suite.
    - [ ] Deploy changes.
    - [ ] Trigger a manual run of the Cloud Run Job (or script) to verify end-to-end multi-coordinate fetching.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Deployment & Verification' (Protocol in workflow.md)
