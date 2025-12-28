# Plan: Fix Bulk Update Functionality

This plan outlines the steps to diagnose and fix the bulk update feature, ensuring data is correctly persisted to BigQuery and the user receives accurate feedback.

## Phase 1: Investigation and Root Cause Analysis [checkpoint: f6a5a99]
- [x] Task: Conductor - Create a reproduction unit test for `bulk_update_reviews` to confirm the failure. f6a5a99
- [x] Task: Conductor - Add detailed logging to `bulk_update_reviews` in `bq_utils.py` to identify the failing step (upload, MERGE, or cleanup). f6a5a99
- [x] Task: Conductor - Inspect the CSV parsing logic in `st_app.py` to ensure the DataFrame passed to the backend is not empty or malformed. f6a5a99
- [x] Task: Conductor - User Manual Verification 'Phase 1: Investigation and Root Cause Analysis' (Protocol in workflow.md) f6a5a99

## Phase 2: Backend Logic Fix [checkpoint: 3f8971d]
- [x] Task: Conductor - Write failing (red) tests for `bulk_update_reviews` covering the identified root cause. 3f8971d
- [x] Task: Conductor - Implement the fix in `bq_utils.py` to ensure the `MERGE` operation succeeds. 3f8971d
- [x] Task: Conductor - Verify the fix by running the new tests and ensuring they pass (green). 3f8971d
- [x] Task: Conductor - User Manual Verification 'Phase 2: Backend Logic Fix' (Protocol in workflow.md) 3f8971d

## Phase 3: UI Enhancement and Reporting
- [ ] Task: Conductor - Modify `bulk_update_reviews` to return the number of affected rows (if available) or a success boolean.
- [ ] Task: Conductor - Update `st_app.py` to display a success message with the count of updated rows.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI Enhancement and Reporting' (Protocol in workflow.md)

## Phase 4: Final Verification and Deployment
- [ ] Task: Conductor - Run the full automated test suite to ensure no regressions.
- [ ] Task: Conductor - Deploy the fixed application to Cloud Run.
- [ ] Task: Conductor - Final Manual Verification in the redeployed environment.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Final Verification and Deployment' (Protocol in workflow.md)
