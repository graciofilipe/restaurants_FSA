# Specification: Comprehensive Local Authority Exclusion List

## 1. Overview
The user requires the "Local Authorities to Exclude" filter in the Streamlit application to be fully comprehensive, reflecting all distinct Local Authorities present in the BigQuery Master Table. Currently, the list is loaded once and stored in the session state, which may lead to stale or incomplete data if the database updates. This track will implement a refresh mechanism and add validation logging to ensure data integrity.

## 2. Functional Requirements

### 2.1 Backend (BigQuery Utilities)
- **Validation Logging:** Update `get_distinct_local_authorities` in `app/services/bq_utils.py` to log the total count of distinct local authorities fetched from BigQuery.
  - *Format:* "Fetched {count} distinct Local Authorities from {table}."

### 2.2 Frontend (Streamlit App)
- **Refresh Mechanism:** Add a "Refresh Authorities" button near the "Exclude Local Authorities" multiselect widget in `app/ui/st_app.py`.
- **Behavior:**
  - When clicked, this button must clear the cached `la_options` from `st.session_state`.
  - It should trigger a re-fetch of the distinct local authorities from BigQuery.
  - A success message (e.g., "Refreshed list. Found X authorities.") should be displayed temporarily.

## 3. Non-Functional Requirements
- **Performance:** The refresh operation should use a spinner (`st.spinner`) to indicate activity, as fetching distinct values from a large table may take a moment.
- **Usability:** The refresh button should be unobtrusive but easily accessible within the "Review Parameters" sidebar section.

## 4. Acceptance Criteria
- [ ] Clicking "Refresh Authorities" updates the list of available options in the "Exclude Local Authorities" dropdown.
- [ ] Application logs confirm the exact number of authorities fetched matches the result of a `SELECT COUNT(DISTINCT localauthorityname)` query in BigQuery.
- [ ] The list persists in `st.session_state` between other interactions (like changing filters) to prevent unnecessary re-fetching.

## 5. Out of Scope
- Fetching the list from the external FSA API (we are strictly using the BigQuery Master Table as the source of truth).
