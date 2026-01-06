# Specification: Codebase Refactor and Simplification

## 1. Overview
The goal of this track is to reorganize the current flat project structure into a modular, functional architecture. This involves grouping files by their role (UI, Services, Core Logic, Scripts) to improve readability, maintainability, and scalability. Additionally, existing code will be refactored to separate concerns (specifically UI vs. Business Logic) and standardize coding style.

## 2. Functional Requirements

### 2.1 Directory Structure Reorganization
- **Goal:** Implement a "Functional Layering" folder structure.
- **New Structure:**
    - `app/ui/`: Streamlit-specific code and view components.
    - `app/services/`: External integrations (API clients, BigQuery utils).
    - `app/core/`: Core business logic and data processing.
    - `scripts/`: Maintenance and one-off execution scripts (currently in root).
- **Action:** Move existing files to these new locations.

### 2.2 Component Refactoring
- **UI Logic (`st_app.py`):**
    - Isolate Streamlit UI definitions from data fetching and processing logic.
    - Move business logic to `app/core/` or `app/services/` as appropriate.
- **Service Layer (`api_client.py`, `bq_utils.py`):**
    - Refactor into modular service classes or functions within `app/services/`.
    - Improve error handling patterns.
- **Scripts:**
    - Move `bq_scripts.py`, `bigQuery_scripts.txt`, and other root-level scripts to `scripts/`.

### 2.3 Code Style & Cleanup
- **Style Standard:** Follow the Google Python Style Guide (as per `conductor/code_styleguides/python.md`).
- **Enforcement:** Best effort manual application (no new strict CI/CD tooling).
- **Cleanup:** Remove obsolete files and update all import statements to reflect the new structure.

## 3. Non-Functional Requirements
- **Maintainability:** The new structure should make it easier for developers to locate code.
- **Backwards Compatibility:** The application functionality must remain unchanged. The entry point for the Streamlit app might change (location), but the app behavior should be identical.

## 4. Acceptance Criteria
- [ ] The project root is clean, containing only configuration files (`Dockerfile`, `requirements.txt`, etc.) and the top-level `app` and `scripts` directories.
- [ ] All Python imports are updated and resolved correctly.
- [ ] The Streamlit application runs successfully from the new structure.
- [ ] `st_app.py` (or its successor) contains primarily UI rendering code.
- [ ] Business logic is covered by tests (existing or adapted).

## 5. Out of Scope
- Adding new product features.
- Implementing strict automated linting/formatting pipelines (e.g., pre-commit hooks).
