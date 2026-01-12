# Plan: Enforce Selected Rows for Insights

## Phase 1: Agent Research Refactoring [checkpoint: d8ce126]
- [x] Task: Clean up `app/ui/agent_research.py` 03feec4
    - [x] Subtask: Remove `render_agent_research_tab` "Target Mode" selection logic.
    - [x] Subtask: Remove logic and UI code for "Batch (All Filtered)" and "Manual List" modes.
    - [x] Subtask: Implement "Selected Rows" as the default and only behavior.
    - [x] Subtask: Implement UI state validation (disable button if `selected_rows` is empty).
- [x] Task: Conductor - User Manual Verification 'Phase 1: Agent Research Refactoring' (Protocol in workflow.md)

## Phase 2: Gemini Analysis Backend Update
- [ ] Task: Modify SQL generation in `scripts/bq_scripts.py`
    - [ ] Subtask: Update `get_gemini_enrichment_script` (or equivalent) to accept an optional list of `fhrsid`s.
    - [ ] Subtask: Add a `WHERE T.fhrsid IN (...)` clause to the `UPDATE` statement (and/or the source selection CTE) to restrict processing to the provided IDs.
- [ ] Task: Update Python Wrapper in `app/services/bq_utils.py`
    - [ ] Subtask: Modify `execute_gemini_enrichment` signature to accept `fhrsids: List[str]`.
    - [ ] Subtask: Pass the list of IDs to the SQL script formatting function.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Gemini Analysis Backend Update' (Protocol in workflow.md)

## Phase 3: Gemini Analysis UI Update
- [ ] Task: Update `app/ui/st_app.py`
    - [ ] Subtask: Locate `tab_gemini` rendering block.
    - [ ] Subtask: Pass `selected_rows` data to the logic.
    - [ ] Subtask: Disable "Run Gemini Analysis" button if `selected_rows` is empty.
    - [ ] Subtask: Extract `fhrsid` list from selection and pass to `execute_gemini_enrichment`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Gemini Analysis UI Update' (Protocol in workflow.md)

## Phase 4: Deployment & Verification
- [ ] Task: Deploy to Cloud Run
    - [ ] Subtask: Run `gcloud builds submit` or equivalent deployment command.
- [ ] Task: Production Verification
    - [ ] Subtask: Open production URL.
    - [ ] Subtask: Select specific rows.
    - [ ] Subtask: Verify "Agent Research" button enables/disables correctly.
    - [ ] Subtask: Verify "Gemini Analysis" button enables/disables correctly.
    - [ ] Subtask: Run "Gemini Analysis" on a small selection and verify only those rows are updated in BigQuery/UI.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Deployment & Verification' (Protocol in workflow.md)
