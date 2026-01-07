# Specification: Vertex AI Agent with Google Maps Grounding Prototype

## 1. Overview
This track focuses on creating a standalone prototype of an AI Agent using Google Cloud's Vertex AI Agent Engine. The core capability to be developed and verified is "Grounding with Google Maps," enabling the agent to answer questions about real-world locations (restaurants) using real-time data.

This is a **technology verification** step. The goal is to prove the infrastructure and API configuration before integrating this capability into the main application.

## 2. Functional Requirements

### 2.1 Vertex AI Agent Configuration
- **Agent Creation:** A new Agent must be configured in Vertex AI Agent Engine.
- **Grounding Source:** The agent must be explicitly configured to use "Google Maps" as a grounding source.
- **Model:** Use a Gemini model capable of handling grounding (e.g., `gemini-1.5-flash` or `gemini-1.5-pro` as appropriate for the region/availability).

### 2.2 Interactive CLI Client
- **Script:** Develop a Python script (e.g., `scripts/prototype_maps_agent.py`) to interact with the agent.
- **Input:** The script should accept natural language user queries via the terminal (standard input).
- **Output:** The script should print the agent's natural language response to the terminal (standard output).
- **Loop:** The script should run in a loop, allowing for a continuous conversation until the user exits.

### 2.3 Verification Capabilities
- **Real-time Accuracy:** The agent must provide answers that reflect current data from Google Maps (e.g., address, rating, opening hours).
- **Console Access:** The setup must allow the user to also inspect/test the agent directly within the Vertex AI Agent Engine console.

## 3. Non-Functional Requirements
- **Environment:** The script must run within the project's existing Python environment (`.venv`).
- **Authentication:** Use Application Default Credentials (ADC) or the project's existing service account setup for authentication.
- **Dependencies:** Add any necessary Google Cloud libraries to `requirements.txt` (e.g., `google-cloud-aiplatform`).

## 4. Acceptance Criteria
1.  **Dependencies Installed:** All required Python libraries are installed and documented in `requirements.txt`.
2.  **CLI Interaction:** The user can run `python scripts/prototype_maps_agent.py`, ask "Where is Nando's in London?", and receive a coherent, factual response based on Maps data.
3.  **Grounding Verification:** The response (or logs) indicates that Google Maps data was actually used (e.g., via citation or explicit grounding metadata, if available in the API response).
4.  **No Main App Impact:** The main Streamlit application (`st_app.py`) remains unchanged and functional.

## 5. Out of Scope
- Integration with the main Streamlit web UI.
- Cross-referencing Maps data with the BigQuery FSA hygiene database.
- Deployment of the agent as a Cloud Run service (this is a local script only).
