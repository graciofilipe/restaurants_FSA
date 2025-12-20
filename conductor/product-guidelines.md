# Product Guidelines: FSA API Explorer

## Tone and Voice
- **Technical & Concise:** The application should use precise, data-oriented language. Documentation and UI labels should be direct and focused on the technical operation of fetching and comparing data.
- **Developer-Friendly:** Since the primary user is a developer, use standard technical terminology (e.g., "Upsert," "Delta," "Schema Validation") where appropriate.

## User Experience (UX) and Visual Identity
- **Clean & Modern Layout:** Prioritize high readability. Use a modern Streamlit aesthetic with ample whitespace, clear typography, and a structured layout that doesn't overwhelm the user.
- **Sectioned Information:** Organize the UI into distinct phases (Input, Fetching, Discovery Results, Syncing) to guide the user through the discovery process.

## Data Presentation Strategy
- **New Establishment Focus (Default):** The primary view must filter results to show *only* new discoveries. This minimizes noise and highlights the "discovery" goal.
- **On-Demand Context:** Provide toggles or separate tabs to view the broader "master list" or updated records for existing establishments only when requested.
- **Action-Oriented Rows:** Each restaurant entry should prioritize the data points most useful for manual research (Name, Address, Rating, Link to Research).
