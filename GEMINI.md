# FSA API Explorer Project Context

## Project Overview
This project is a Streamlit-based web application designed to fetch, analyze, and store food hygiene rating data from the Food Standards Agency (FSA) API. It integrates with Google BigQuery for data persistence, allowing users to maintain a master list of restaurant ratings, identify new establishments, and update existing records.

## Architecture & Tech Stack
*   **Frontend/UI:** Streamlit (`st_app.py`)
*   **Backend Logic:** Python 3.11
*   **Data Source:** FSA API (UK Food Hygiene Ratings)
*   **Data Storage:** Google BigQuery
*   **Containerization:** Docker
*   **CI/CD:** Google Cloud Build (`cloudbuild.yaml`) & Cloud Run

## Key Files and Directories

### Source Code
*   **`st_app.py`**: The main entry point for the Streamlit application. It handles the UI layout, user inputs (coordinates, BigQuery paths), and orchestrates the data fetching and storage workflows.
*   **`api_client.py`**: Contains the `fetch_api_data` function to interact with the FSA API.
*   **`bq_utils.py`**: A utility module for all BigQuery interactions. It includes functions for:
    *   `load_all_data_from_bq`: Reading master data.
    *   `write_to_bigquery`: Overwriting tables.
    *   `append_to_bigquery`: Appending new records.
    *   `update_rows_in_bigquery`: updating specific rows.
    *   `sanitize_column_name`: ensuring DataFrame columns match BigQuery schema requirements.
*   **`data_processing.py`**: (Inferred) Handles data normalization, comparison between new API results and existing master data to identify new restaurants.

### Configuration & Deployment
*   **`Dockerfile`**: Defines the container image based on `python:3.11-slim`.
*   **`cloudbuild.yaml`**: CI/CD configuration for Google Cloud Build. It installs dependencies, runs tests, builds the Docker image, and deploys to Cloud Run.
*   **`requirements.txt`**: Python package dependencies (e.g., `streamlit`, `google-cloud-bigquery`, `pandas`, `pytest`).

### Testing
*   **`test_bq_utils.py`**: Unit tests for BigQuery utility functions, utilizing `unittest.mock` to simulate BigQuery client interactions.
*   **`test_st_app.py`**: (Inferred) Tests for the Streamlit application logic.

## Setup and Usage

### Prerequisites
*   Python 3.7+ (3.11 recommended)
*   Google Cloud SDK (gcloud) configured with Application Default Credentials (ADC) for BigQuery access.

### Installation
1.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Locally
To start the Streamlit app:
```bash
streamlit run st_app.py
```
The app will be accessible at `http://localhost:8501`.

### Running Tests
To execute the test suite:
```bash
pytest
```

## Development Conventions

*   **BigQuery Schemas:** Schemas are defined explicitly in the code (e.g., inside `bq_utils.py` or test files) rather than in external SQL/JSON files.
*   **Column Sanitization:** The project uses a strict column sanitization function (`sanitize_column_name`) to ensure BigQuery compatibility (lowercase, underscores, no special chars).
*   **Type Safety:** Type hinting (e.g., `List[Dict[str, Any]]`) is used throughout the codebase.
*   **Testing:** Heavy reliance on mocking for external services (FSA API, BigQuery) to ensure isolated unit tests.
*   **Error Handling:** Custom exceptions like `BigQueryExecutionError` are defined to handle specific failure modes.

## Cloud Deployment
The application is deployed to Google Cloud Run via Cloud Build. The build steps are:
1.  Install dependencies.
2.  Run tests (`pytest`).
3.  Build Docker image.
4.  Push image to Google Container Registry (GCR).
5.  Deploy to Cloud Run (`restaurants-fsa` service in `europe-west2`).

## Recent Updates
*   **Agent Interaction Upgrade:** Switched from the legacy `ReasoningEngine` client to the cloud-native `vertexai.Client` and `AgentEngine` SDK to correctly handle ADK-based agents and async stream queries. This resolved an `AttributeError` in production.
*   **Gemini 3 Integration Fixes:** Updated `bq_scripts.py` to use the correct `model_params` structure for the Vertex AI GenerateContent API.
    *   Corrected `tools` format to `[{"googleSearch": {}}]`.
    *   Added explicit `safetySettings` with `threshold: "OFF"`.
    *   Adjusted `generationConfig` (temperature 0.6, topP 0.72, maxOutputTokens 65535).
    *   Removed unsupported `thinkingConfig` to resolve API 400 errors.