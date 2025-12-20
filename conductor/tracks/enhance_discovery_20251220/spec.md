# Specification: Enhance Discovery Workflow

## Goal
Improve the user's ability to discover and research new restaurants by providing direct access to external research tools (Google Maps) and ensuring the "new restaurant" identification logic is reliable.

## Features
1.  **Google Maps Integration:**
    -   Construct search URLs for each restaurant using its name, address, and postcode.
    -   Example URL format: `https://www.google.com/maps/search/?api=1&query={Name}+{Address}+{Postcode}`.
    -   Ensure robust URL encoding to handle special characters in restaurant names.

2.  **UI Enhancements:**
    -   Update the "New Discoveries" table in Streamlit to include a clickable "Research on Maps" link.
    -   Ensure the link opens in a new tab for seamless browsing.
    -   Prioritize the display of this link alongside the Restaurant Name and Rating.

3.  **Robust Delta Logic:**
    -   Review current `data_processing.py` logic for identifying new restaurants.
    -   Enhance comparison to handle minor data inconsistencies (e.g., casing, whitespace) to avoid false positives (identifying a known restaurant as new).
    -   Ensure the comparison logic is efficient enough to run on potentially large master lists.

## Technical Considerations
-   **URL Encoding:** Python's `urllib.parse.quote_plus` should be used for query parameters.
-   **Streamlit Display:** Use `st.dataframe` with `st.column_config.LinkColumn` for a clean, interactive table.
-   **BigQuery Schema:** No schema changes are expected; logic will operate on DataFrames before/after BigQuery interaction.
