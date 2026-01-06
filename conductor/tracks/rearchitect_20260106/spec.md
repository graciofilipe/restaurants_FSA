# Specification: Re-architect for Automated Ingestion and Controlled Review

## 1. Overview
This track re-architects the application from a manual, user-driven fetch process to a hybrid model:
1.  **Automated Ingestion:** A weekly background process automatically fetches new restaurant data from the FSA API and updates the BigQuery master table.
2.  **Controlled Review UI:** The Streamlit interface is refocused on analyzing and exporting this pre-fetched data. It starts in an "empty" state, requiring the user to parameterize their review scope (date, status, exclusions) before displaying results or enabling Gemini analysis.

## 2. Functional Requirements

### 2.1 Automated Ingestion Service
- **Mechanism:** Google Cloud Scheduler triggering a **Cloud Run Job**.
- **Command:** The job executes a Python script (e.g., `python -m app.cron.fetch_weekly`).
- **Configuration (BigQuery Table):**
    - A new table (e.g., `config_search_params`) will store the execution parameters.
    - **Schema:**
        - `coordinates` (STRING): "lat,long" pairs (e.g., "51.5,-0.1").
        - `max_results` (INTEGER): API fetch limit per coordinate.
        - `target_bq_table` (STRING): Full path to the master table (e.g., `project.dataset.table`).
- **Process:**
    - Read configuration from `config_search_params`.
    - Iterate through configured coordinates.
    - Fetch data from FSA API.
    - Upsert into `target_bq_table`.
    - Set `manual_review` status to `"not reviewed"` for new records.
    - Set `first_seen` to the current timestamp.

### 2.2 On-Demand Analysis & Export UI (Streamlit)
- **Initial State:** The main dashboard is empty or shows a "Configure Review" sidebar/section. No data is loaded by default.
- **User Parameters:**
    - **Minimum "First Seen" Date:** Date picker.
    - **Review Status:** Multi-select (Default: "not reviewed", but selectable).
    - **Exclude Local Authorities:** Multi-select (Populated dynamically from distinct values in the master table).
- **"Load Results" Action:**
    - User clicks a "Load Data" button after parameterizing.
    - **Result:** Displays a DataFrame/Metrics of records matching the criteria.
- **Refined Gemini Analysis:**
    - **Trigger:** "Run Gemini Analysis" button (enabled only after "Load Results").
    - **Logic:** Sends *only* the currently loaded/filtered records to the Gemini enrichment process.
- **Refined Export:**
    - **Trigger:** "Export CSV" button (enabled only after "Load Results").
    - **Logic:** Exports the currently loaded/filtered DataFrame.

### 2.3 Cleanup
- Remove the legacy manual "Fetch Data" input fields from the UI.

## 3. Data Model Changes
- **New Table:** `config_search_params` as defined above.

## 4. Acceptance Criteria
- [ ] `config_search_params` table created and populated with initial config.
- [ ] `app.cron.fetch_weekly` script implemented and verified to fetch/store data based on the config table.
- [ ] Cloud Run Job configuration (or instructions) prepared.
- [ ] Streamlit UI defaults to an "empty" state waiting for user input.
- [ ] User can successfully filter data by Date, Status, and Excluded Authorities.
- [ ] Gemini Analysis and CSV Export actions operate *strictly* on the filtered dataset defined by the user.

## 5. Out of Scope
- Building a UI to edit the `config_search_params` table.
