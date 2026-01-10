# Tech Stack: FSA API Explorer

## Core Technologies
- **Programming Language:** Python 3.11
- **Web Framework:** Streamlit (UI/Frontend)
- **Data Analysis:** Pandas
- **API Client:** Requests (fetching FSA API data)

## Infrastructure & Data Storage
- **Primary Database:** Google BigQuery (Data warehouse for master list)
- **Data Transfer:** `pandas-gbq` and `google-cloud-bigquery`
- **Cloud Platform:** Google Cloud Platform (GCP)

## Artificial Intelligence
- **LLM/Generative AI:** Vertex AI & Google GenAI (Gemini 2.5 Flash for data processing tasks and Agent).
- **Agent Framework:** Google ADK (Agent Development Kit) & Vertex AI Agent Engine.
- **Agent Interaction:** Vertex AI SDK for Python (Cloud-native Client).
- **State Management:** Streamlit Session State (for managing batch processing results and UI visibility).

## DevOps & Tools
- **Containerization:** Docker
- **CI/CD:** Google Cloud Build
- **Runtime:** Google Cloud Run
- **Testing:** Pytest
- **Environment Management:** `requirements.txt` and Shell scripts (`envs.sh`)
