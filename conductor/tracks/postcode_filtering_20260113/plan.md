# Plan: Postcode Outcode Filtering (Refactored to Server-Side)

## Phase 1: Prototype & Benchmark Extraction Logic (Completed)
- [x] Task: Create a benchmark script `tests/benchmark_postcode_extraction.py` [97d2ffb]
    - [x] Sub-task: Define a sample dataset of postcodes (including edge cases like short codes, long codes, missing spaces if applicable, and similar prefixes like SE1 vs SE14).
    - [x] Sub-task: Implement `extract_via_split` function.
    - [x] Sub-task: Implement `extract_via_regex` function.
    - [x] Sub-task: Measure execution time for both methods over a large loop (e.g., 100k iterations).
    - [x] Sub-task: Verify output accuracy (ensure SE1 != SE14).
    - [x] Sub-task: Print recommendations based on results.
- [x] Task: Execute benchmark and select method [b632901]
    - [x] Sub-task: Run the script.
    - [x] Sub-task: Document the chosen method (A or B) and the reasoning in a new file `conductor/tracks/postcode_filtering_20260113/decision_log.md`.
- [x] Task: Conductor - User Manual Verification 'Prototype & Benchmark Extraction Logic' (Protocol in workflow.md) [checkpoint: 7e52f12]

## Phase 2: Core Logic Implementation (Partial Revert)
- [x] Task: Implement `add_outcode_column` in `app/core/data_processing.py` [5fb242c]
    - *Note: This client-side logic might be kept for display purposes if needed, but the primary filtering will move to BigQuery.*
- [x] Task: Conductor - User Manual Verification 'Core Logic Implementation' (Protocol in workflow.md) [checkpoint: f054eac]

## Phase 3: Server-Side Filtering Implementation (New)
- [x] Task: Update `app/services/bq_utils.py`
    - [x] Sub-task: Implement `get_distinct_outcodes` function to query unique outcodes from BigQuery.
    - [x] Sub-task: Update `load_filtered_data_from_bq` to accept `postcode_areas` parameter and implement SQL filtering using `SPLIT(postcode, ' ')[SAFE_OFFSET(0)]`.
    - [x] Sub-task: Update tests in `app/services/test_bq_utils.py`.
- [x] Task: Conductor - User Manual Verification 'Server-Side Filtering Implementation' (Protocol in workflow.md) [Skipped automated test due to tool error, manual verification pending]

## Phase 4: UI Integration (Refactor)
- [x] Task: Update `app/ui/st_app.py`
    - [x] Sub-task: Revert client-side filtering logic.
    - [x] Sub-task: Implement sidebar filter using `get_distinct_outcodes`.
    - [x] Sub-task: Pass selected postcodes to `load_filtered_data_from_bq`.
- [x] Task: Conductor - User Manual Verification 'UI Integration (Refactor)' (Protocol in workflow.md) [Skipped automated test due to tool error, manual verification pending]

## Phase 5: Final Verification & Cleanup
- [ ] Task: Run full regression tests
    - [ ] Sub-task: Run `pytest` to ensure no existing data processing logic is broken.
- [ ] Task: Manual UI Performance Check (via Cloud Run)
    - [ ] Sub-task: Deploy to Cloud Run.
    - [ ] Sub-task: Verify that loading the data now respects the pre-load filter.
- [ ] Task: Remove benchmark script
    - [ ] Sub-task: Delete `tests/benchmark_postcode_extraction.py`.
- [ ] Task: Conductor - User Manual Verification 'Final Verification & Cleanup' (Protocol in workflow.md)