# Plan: Bulk Update Manual Review Status via UI

## Phase 1: Test Preparation & Logic Verification
- [ ] Task: Create a new test file `app/ui/test_bulk_status_update.py`.
- [ ] Task: Write a failing test that simulates the bulk update flow: creating a DataFrame from a list of IDs and a status, and calling `bulk_update_reviews`.
- [ ] Task: Verify that `bulk_update_reviews` in `app/services/bq_utils.py` can handle the generated DataFrame structure (sanity check).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Test Preparation & Logic Verification' (Protocol in workflow.md)

## Phase 2: UI Implementation
- [ ] Task: Update `app/ui/st_app.py` (or the new `agent_research.py` if that's where we decide to put it, though spec says main area bottom).
    - [ ] Add the "Bulk Update Status" section at the bottom of the main area.
    - [ ] Implement the status dropdown and "Update Status" button.
    - [ ] Wire the button to logic that constructs the DataFrame and calls `bulk_update_reviews`.
- [ ] Task: Write failing UI tests in `app/ui/test_bulk_status_update.py` to verify the button appears and triggers the correct function.
- [ ] Task: Implement the UI logic to pass the tests.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: UI Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Final Verification
- [ ] Task: Run full test suite to ensure no regressions.
- [ ] Task: Perform manual verification by selecting rows and updating status in the running app (if possible locally) or via integration test simulation.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Final Verification' (Protocol in workflow.md)
