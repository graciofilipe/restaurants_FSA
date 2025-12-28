# Specification: Fix Bulk Update Functionality

## Overview
The "Bulk Update Manual Reviews" feature is currently failing to update the BigQuery table. While the UI allows file uploads and button clicks, the data persistence layer is not functioning as intended. This track involves root-cause analysis and a fix to ensure that `manual_review` statuses are correctly updated in BigQuery and reported in the UI.

## Functional Requirements
- **Root Cause Analysis:**
    - Investigate the `bulk_update_reviews` function in `bq_utils.py` to identify failures in temporary table creation, data upload, or the `MERGE` operation.
    - Verify that `st_app.py` correctly parses the uploaded CSV and passes a valid DataFrame to the backend.
- **Backend Fix:**
    - Ensure the BigQuery `MERGE` query correctly matches `fhrsid` and updates the `manual_review` column.
    - Improve error handling to capture and display specific BigQuery or Python exceptions.
- **UI Enhancement:**
    - Report the number of rows successfully updated in the Streamlit interface.
    - Provide a clear success message or a detailed error message if the operation fails.

## Non-Functional Requirements
- **Reliability:** The update operation must be atomic; it should either succeed fully or fail with a clear error without corrupting data.
- **Observability:** Implement sufficient logging (via `st.error` or standard logging) to diagnose future failures.

## Acceptance Criteria
- [ ] The `manual_review` column in the BigQuery master table is correctly updated when a valid CSV is uploaded.
- [ ] The UI displays a success message including the count of updated rows (e.g., "Successfully updated 15 rows.").
- [ ] The application remains responsive and handles malformed CSVs gracefully with an error message.
- [ ] Automated tests for `bulk_update_reviews` pass and cover the fix.
- [ ] **Verification:** The fix has been verified by the user in the redeployed application.

## Out of Scope
- Changing the schema of the master table.
- Implementing bulk updates for columns other than `manual_review`.
