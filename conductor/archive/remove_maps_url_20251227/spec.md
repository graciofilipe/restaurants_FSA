# Specification: Remove Maps URL Generation

## Overview
This track involves removing the "Maps Link" generation feature from the FSA API Explorer. A bug was reported where missing data caused a `TypeError` during URL construction. Instead of fixing the bug, the user has requested to remove this component entirely to simplify the application.

## Functional Requirements
- **Logic Removal:**
    -   Remove the `generate_maps_url` function from the codebase.
    -   Delete the `utils/url_generator.py` file since it will be empty.
    -   Remove all calls to `generate_maps_url` in `data_processing.py`.
- **UI/Data Presentation:**
    -   Ensure the "Maps Link" column is no longer added to the restaurant dataframes.
    -   Remove any references to the "Maps Link" in `st_app.py` (e.g., in `st.column_config.LinkColumn`).

## Non-Functional Requirements
- **Stability:** The removal of this feature must not impact the core functionality of fetching FSA data or storing it in BigQuery.
- **Cleanup:** Unused tests related to URL generation should also be removed.

## Acceptance Criteria
- [x] `utils/url_generator.py` is deleted.
- [x] No references to `generate_maps_url` remain in `data_processing.py`.
- [x] `st_app.py` no longer displays or configures a "Maps Link" column.
- [x] Tests related to URL generation (`test_url_generator.py`) are deleted.
- [x] The application runs successfully and fetches data without crashing.

## Out of Scope
- Adding alternative research links (TripAdvisor, Yelp, etc.).
- Modifying the BigQuery schema (the "Maps Link" column was likely a runtime addition, but its absence in BQ should be confirmed).
