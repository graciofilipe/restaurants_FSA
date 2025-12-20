# Track Plan: Enhance Discovery Workflow

## Phase 1: Google Maps Link Generation
- [x] Task: Create `utils/url_generator.py` and implement `generate_maps_url` function with TDD. aa32c5b
    - [ ] Subtask: Write failing tests for `generate_maps_url` (check correct formatting and encoding of names/addresses).
    - [ ] Subtask: Implement `generate_maps_url` using `urllib.parse`.
    - [ ] Subtask: Refactor and verify coverage.
- [x] Task: Integrate `generate_maps_url` into `data_processing.py`. 63b272b
    - [ ] Subtask: Write failing tests for data processing enhancement (verifying the new column is added).
    - [ ] Subtask: Update `data_processing.py` to apply the function and add a 'Maps Link' column to the results DataFrame.
    - [ ] Subtask: Verify tests pass.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Google Maps Link Generation' (Protocol in workflow.md)

## Phase 2: Robust Delta Logic
- [ ] Task: Audit and Improve Delta Logic in `data_processing.py`.
    - [ ] Subtask: Create a reproduction test case simulating edge cases (case sensitivity, whitespace differences) where "existing" restaurants are incorrectly flagged as "new".
    - [ ] Subtask: Implement robust normalization (lowercasing, stripping whitespace) in the comparison logic.
    - [ ] Subtask: Verify tests pass and false positives are eliminated.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Robust Delta Logic' (Protocol in workflow.md)

## Phase 3: UI Integration
- [ ] Task: Update Streamlit UI to Display Links.
    - [ ] Subtask: Review `st_app.py` and identify the dataframe display section.
    - [ ] Subtask: Update `st_app.py` to use `st.column_config.LinkColumn` for the 'Maps Link' column, ensuring it renders as a clickable link named "Search Maps".
    - [ ] Subtask: Verify locally that the links work and open in a new tab.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: UI Integration' (Protocol in workflow.md)
