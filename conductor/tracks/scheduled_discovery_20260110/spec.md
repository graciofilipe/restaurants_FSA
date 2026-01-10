# Specification: Scheduled Discovery & Config Update

## Overview
This track focuses on operationalizing the data discovery process. Currently, data fetching is a manual or ad-hoc process. We will deploy a Google Cloud Run Job to automate this on a weekly schedule. Additionally, we will provide a mechanism to update the geographic search parameters stored in BigQuery to ensure the discovery covers the user's areas of interest.

## Functional Requirements

### 1. Search Configuration Management
- **Script:** A Python script (`scripts/update_search_config.py`) to interface with the `config_search_params` BigQuery table.
- **Capabilities:**
    - List current configuration.
    - Add new search points (Lat/Lon/Max Results).
    - (Optional) Remove existing points.
- **Data Source:** User-provided coordinates.

### 2. Automated Job Deployment
- **Resource:** Google Cloud Run Job named `fetch-weekly`.
- **Image:** The existing production image (`gcr.io/filipegracio-ai-learning/python-app:latest`).
- **Command:** `python -m app.cron.fetch_weekly`.
- **Environment:** Must have access to BigQuery (via default Service Account).

### 3. Scheduling
- **Resource:** Google Cloud Scheduler.
- **Schedule:** Weekly (e.g., `0 9 * * 1` for Monday 9am).
- **Target:** The `fetch-weekly` Cloud Run Job.

## Non-Functional Requirements
- **Observability:** Job execution must be visible in Cloud Run logs.
- **Idempotency:** The fetch script is already designed to upsert/merge, preventing duplicate data issues.

## User Inputs Needed
- **New Coordinates:** The user must provide the specific Lat/Lon pairs to be added to the search configuration.
