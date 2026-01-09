# Specification: Agent Insights Generation

## Overview
This track introduces the ability for users to generate and store insights for specific restaurants using the recently deployed Google Maps agent. Users will be able to select restaurants from their filtered results and trigger a research process that fetches cuisine types, review counts, and average ratings from Google Maps, storing the data in a dedicated BigQuery table.

## Functional Requirements
1.  **Restaurant Selection Modes:**
    -   **Interactive Selection:** (Existing) Select rows via checkboxes in the results table.
    -   **Batch Mode:** Process all restaurants currently loaded in the "Review Queue" (filtered view).
    -   **Manual List:** specific `fhrsid`s provided via a text area (comma or newline separated).
2.  **Trigger Mechanism:**
    -   Add a "Generate Agent Insights" button in the "Analysis/Insights" section.
    -   Include a selector (Radio or Selectbox) to choose the "Target Mode" (Selection, Batch, Manual).
    -   The button should be enabled if valid targets are found for the selected mode.
3.  **Agent Orchestration:**
    -   For each target restaurant, call the Google Maps agent (`app/maps_agent/agent.py`).
    - The prompt to the agent must specifically request:
        - Type of restaurant (cuisine).
        - Number of reviews.
        - Average rating.
4.  **Data Processing & Parsing:**
    - Capture the raw text response from the agent.
    - Extract structured data (cuisine, review count, average rating) from the response.
5.  **Data Persistence (BigQuery):**
    - Create/Manage a table named `restaurant_agent_insights`.
    - Schema: `fhrsid` (STRING/INT64), `raw_insight` (STRING), `cuisine_type` (STRING), `review_count` (INT64/FLOAT64), `average_rating` (FLOAT64), `updated_at` (TIMESTAMP).
    - **Logic:** Upsert. If a record for a specific `fhrsid` already exists, overwrite it with the new findings and update the `updated_at` timestamp.

## Non-Functional Requirements
- **Performance:** Process requests in a way that provides feedback to the user (e.g., a progress bar) since agent calls may take time.
- **Robustness:** Handle cases where the agent cannot find a restaurant or returns an unexpected format.

## Acceptance Criteria
- [ ] Users can select multiple restaurants using checkboxes in the UI.
- [ ] Clicking "Generate Agent Insights" successfully calls the Maps agent for each selection.
- [ ] Data is correctly parsed and stored in the `restaurant_agent_insights` BigQuery table.
- [ ] Repeated requests for the same restaurant update the existing record instead of creating duplicates.
- [ ] The UI provides visual feedback (success messages or progress indicators) during and after processing.
- [ ] The feature is fully verifiable and functional in the deployed Cloud Run environment.

## Out of Scope
- Automated weekly/bulk generation for all new restaurants (this remains a manual user-triggered action for now).
- Displaying the insights back in the main results table (this will be handled in a future track).
