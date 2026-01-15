# Specification: Postcode Outcode Filtering

## 1. Overview
The goal of this track is to introduce a new filtering mechanism to the FSA Restaurant Discovery Tool that allows users to filter restaurants by their postcode "outcode" (the first segment of a UK postcode, e.g., "SE1", "SW16"). This will function similarly to the existing Local Authority and Status filters, enabling more granular geographic discovery.

## 2. Functional Requirements
- **Outcode Extraction:** The system must reliably extract the "outcode" from the full `PostCode` field in the restaurant data.
    - **Logic:** The extraction logic must be strictly "exact match" based. Selecting "SE1" must NOT return results for "SE14".
    - **Implementation:** Two extraction methods (Whitespace Split vs. Regex) will be prototyped and benchmarked.
- **UI Filter Widget:**
    - A new multi-select dropdown widget labeled "Select Postcode Area" (or similar) will be added to the sidebar.
    - The options in this dropdown will be dynamically populated based on the unique outcodes available in the currently loaded dataset.
    - The filter will default to "All" (no selection) or allow clearing the selection to show all records.
- **Data Filtering:**
    - Selecting one or more outcodes will filter the main restaurant list to show only those matching the selected outcodes.
    - This filter will operate in conjunction with existing filters (Local Authority, Status, Date).

## 3. Technical Strategy
- **Extraction Location:** "Runtime (Pandas)". The outcode will be derived in Python code (`data_processing.py`) after data is fetched from BigQuery.
- **Performance Consideration:** Latency will be monitored. If runtime extraction proves too slow, persistent storage will be considered (out of scope for this track).
- **Validation, Benchmarking & Selection:**
    - **Method A (Whitespace Split):** `postcode.split(' ')[0]`
    - **Method B (Regex):** `re.match(r'^([A-Z]{1,2}[0-9]{1,2}[A-Z]?)', postcode)`
    - **Selection Protocol:** Both methods will be tested for accuracy and execution speed.
    - **Final Code:** Only the selected, optimal method will be integrated into the final application. The rejected method will be discarded.

## 4. Acceptance Criteria
- [ ] A new "Postcode Area" multi-select filter exists in the application sidebar.
- [ ] The filter options are populated with valid outcodes (e.g., "SE1", "E2") from the current data.
- [ ] Selecting "SE1" strictly filters the list to restaurants with postcodes starting with "SE1 " (and does NOT include "SE14").
- [ ] The chosen extraction method is verified for accuracy and performance.
- [ ] The application loads and filters data without significant UI lag (< 1s perceptible delay for filtering).

## 5. Out of Scope
- Persisting the `outcode` column to the BigQuery database.
- Filtering by the second part of the postcode.
- Map-based polygon filtering.
