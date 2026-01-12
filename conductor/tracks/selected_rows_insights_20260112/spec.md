# Specification: Enforce Selected Rows for Insights

## 1. Overview
The goal of this track is to standardize the user interaction for generating insights (both "Gemini Insights" and "Agent Insights") by enforcing a "Selected Rows Only" mechanism. This means users must explicitly select rows in the main data table before triggering any insight generation. The existing "Batch" and "Manual List" modes for Agent Insights will be removed, and the Gemini Insights process will be updated to respect the user's selection rather than running on the entire filtered dataset.

## 2. Functional Requirements

### 2.1 Agent Research Tab
*   **Remove Modes:** Completely remove the "Target Mode" radio button (Selected Rows, Batch, Manual List).
*   **Implicit Mode:** The tab should function as if "Selected Rows" is always active.
*   **Validation:**
    *   If no rows are selected, the "Generate Agent Insights" button must be **disabled (greyed out)**.
    *   Display a clear message or tooltip indicating that row selection is required.
*   **Action:** Clicking the button processes only the selected restaurants using the existing `handle_insight_generation` logic.

### 2.2 Gemini Analysis Tab
*   **Input Scope:** The analysis must run *only* on the rows selected by the user in the main table.
*   **Validation:**
    *   If no rows are selected, the "Run Gemini Analysis" button must be **disabled (greyed out)**.
*   **Backend Logic:**
    *   Modify `execute_gemini_enrichment` (and underlying SQL scripts) to accept a list of `FHRSID`s.
    *   The SQL query must filter the target dataset using these IDs before invoking the Gemini model, ensuring only selected records are processed and paid for.

## 3. Non-Functional Requirements
*   **Performance:** The filtering should happen at the database level (BigQuery) for Gemini Analysis to maintain efficiency.
*   **Code Cleanliness:** unused code for "Batch" and "Manual List" modes in Agent Research should be fully removed, not just commented out.

## 4. Acceptance Criteria
*   [ ] In the "Agent Research" tab, the "Target Mode" selector is gone.
*   [ ] In "Agent Research", the generate button is disabled when 0 rows are selected.
*   [ ] In "Agent Research", clicking generate processes exactly the N selected rows.
*   [ ] In the "Gemini Analysis" tab, the generate button is disabled when 0 rows are selected.
*   [ ] In "Gemini Analysis", triggering the analysis updates *only* the selected records in BigQuery (verified by checking `gemini_insights` column or logs).
*   [ ] The application does not crash or show errors when switching between tabs with/without selections.
*   [ ] The application is deployed and tested in production.

## 5. Out of Scope
*   Changing the actual prompt content for either agent.
*   Changing the output format/schema of the insights.
