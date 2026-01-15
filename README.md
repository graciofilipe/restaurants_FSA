# Food Standards Agency API Explorer

## Project Overview
This project is a Streamlit-based web application designed to fetch, analyze, and store food hygiene rating data from the Food Standards Agency (FSA) API. It integrates with Google BigQuery for data persistence, allowing users to maintain a master list of restaurant ratings, identify new establishments, and update existing records.

## Architecture & Tech Stack
*   **Frontend/UI:** Streamlit (`app/ui/st_app.py`)
*   **Backend Logic:** Python 3.11 (`app/services`, `app/core`)
*   **Data Source:** FSA API (UK Food Hygiene Ratings)
*   **Data Storage:** Google BigQuery
*   **Containerization:** Docker
*   **CI/CD:** Google Cloud Build (`cloudbuild.yaml`) & Cloud Run

## Key Directories
*   **`app/ui/`**: Contains the Streamlit application logic.
*   **`app/services/`**: Business logic and API interactions.
*   **`app/core/`**: Core definitions and configurations.
*   **`conductor/`**: Orchestration logic (if applicable).
*   **`scripts/`**: Utility scripts (e.g., `prototype_maps_agent.py`).

## Setup and Usage

### Prerequisites
*   Python 3.7+ (3.11 recommended)
*   Google Cloud SDK (gcloud) configured with Application Default Credentials (ADC) for BigQuery access.
*   Firebase configuration in `.streamlit/secrets.toml` (for authentication).

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
streamlit run app/ui/st_app.py
```
The app will be accessible at `http://localhost:8501`.

### Running Tests
To execute the test suite:
```bash
pytest app/
```

## AI Agent Prototype (Maps Grounding)
A prototype AI Agent capable of answering questions about restaurants using Google Maps Grounding is available in `app/maps_agent/`.

## Cloud Deployment
The application is deployed to Google Cloud Run via Cloud Build.

> [!IMPORTANT]
> A Cloud Build trigger is configured to automatically build and deploy the application with every commit and push to the repository. **You must commit and push your changes to the remote repository (`main` branch) to see them reflected in the live application.**

The `cloudbuild.yaml` file defines the build steps:
1.  Install dependencies.
2.  Run tests.
3.  Build Docker image.
4.  Push image to Google Container Registry (GCR).
5.  Deploy to Cloud Run (`restaurants-fsa` service).

To deploy manually (if authorized):
```bash
gcloud builds submit --config cloudbuild.yaml .
```
