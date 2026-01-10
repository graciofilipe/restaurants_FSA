# Specification: Bulk Update Manual Review Status via UI

## Overview
This feature allows users to select multiple restaurants from the main results table in the Streamlit UI and update their `manual_review` status in BigQuery in a single batch operation. This eliminates the need for CSV uploads for simple status changes.

## Functional Requirements

### 1. UI - Selection and Action
- **Location:** The bulk update controls must be placed in the main area at the bottom of the page, below the results table and other existing action buttons.
- **Input:**
    - A dropdown (selectbox) to choose the new status (e.g., "pending", "accepted", "rejected", "not reviewed").
    - A primary button labeled "Update Status for Selected Rows".
- **Interaction:**
    - The button should be disabled if no rows are selected.
    - Upon clicking, the update process initiates immediately (no extra confirmation dialog).

### 2. Backend - Data Processing
- **Logic:**
    - When the button is clicked, collect the `fhrsid`s of all selected rows.
    - Create a Pandas DataFrame containing two columns: `fhrsid` and `manual_review` (set to the selected status for all rows).
    - Invoke the existing `bulk_update_reviews` function in `app/services/bq_utils.py` with this DataFrame.

### 3. Feedback
- **Success:** Display a success message indicating the number of rows updated.
- **Failure:** Display an error message if the BigQuery operation fails.
- **State Update:** Trigger a rerun or data reload to reflect the changes in the UI immediately.

## Non-Functional Requirements
- **Performance:** The update should be efficient, leveraging BigQuery's MERGE capability via the existing utility.
- **Code Quality:** Reuse existing `bulk_update_reviews` to maintain a single source of truth for bulk updates.

## Acceptance Criteria
- [ ] Users can select multiple rows in the main table.
- [ ] A "Bulk Update Status" section is visible at the bottom of the main area.
- [ ] Clicking the update button correctly updates the `manual_review` column in BigQuery for all selected `fhrsid`s.
- [ ] The UI refreshes to show the new statuses after the update completes.
- [ ] The implementation uses the existing `bulk_update_reviews` function.
