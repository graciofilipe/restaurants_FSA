# Plan: Postcode Outcode Filtering

## Phase 1: Prototype & Benchmark Extraction Logic
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

## Phase 2: Core Logic Implementation
- [x] Task: Implement `add_outcode_column` in `app/core/data_processing.py` [5fb242c]
    - [x] Sub-task: Create a new test file `app/core/test_postcode_logic.py`.
    - [x] Sub-task: Write failing tests for `add_outcode_column` ensuring it handles the full DataFrame and correctly adds the column using the chosen logic.
    - [x] Sub-task: Implement the function in `data_processing.py` using Pandas vectorization (e.g., `.str.split` or `.str.extract`) for performance.
    - [x] Sub-task: Verify tests pass.
- [ ] Task: Conductor - User Manual Verification 'Core Logic Implementation' (Protocol in workflow.md)

## Phase 3: UI Integration
- [ ] Task: Update `app/ui/st_app.py` to include the new filter
    - [ ] Sub-task: Locate the sidebar filtering section.
    - [ ] Sub-task: Call `data_processing.add_outcode_column` on the loaded dataframe.
    - [ ] Sub-task: Extract unique values from the new `outcode` column for the dropdown options (sorted alphabetically).
    - [ ] Sub-task: Add `st.multiselect` for "Postcode Area".
    - [ ] Sub-task: Implement the filtering logic: if options are selected, filter the dataframe to include only rows where `outcode` is in the selection.
- [ ] Task: Conductor - User Manual Verification 'UI Integration' (Protocol in workflow.md)

## Phase 4: Final Verification & Cleanup
- [ ] Task: Run full regression tests
    - [ ] Sub-task: Run `pytest` to ensure no existing data processing logic is broken.
- [ ] Task: Manual UI Performance Check
    - [ ] Sub-task: Verify that loading the app and applying filters feels responsive.
- [ ] Task: Remove benchmark script
    - [ ] Sub-task: Delete `tests/benchmark_postcode_extraction.py`.
- [ ] Task: Conductor - User Manual Verification 'Final Verification & Cleanup' (Protocol in workflow.md)