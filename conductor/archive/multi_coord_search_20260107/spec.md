# Specification: Multi-Coordinate Search Support

## 1. Overview
The current automated weekly fetch process is limited to a single geographic coordinate pair. This feature extends the system to support a list of coordinate pairs, allowing the application to monitor multiple distinct areas (or a wider grid) for new restaurant openings. Additionally, the fetch limit (`max_results`) will be increased to 5000 to ensure comprehensive data retrieval.

## 2. Functional Requirements

### 2.1 BigQuery Configuration
- **Table Structure:** The `config_search_params` table must be updated or migrated to support **multiple rows**.
- **Schema:** Each row represents a distinct search target containing:
    - `latitude` (FLOAT)
    - `longitude` (FLOAT)
    - `max_results` (INTEGER, default 5000)
    - `radius` (Integer, optional/existing default)

### 2.2 Data Ingestion (`setup_config_table.py`)
- The setup script must be capable of initializing the table with a **list** of coordinate pairs provided by the user.
- Default `max_results` for these entries will be set to **5000**.

### 2.3 Automated Fetch Logic (`app/cron/fetch_weekly.py`)
- **Iteration:** The script must query `config_search_params` and iterate through **all** returned rows.
- **Execution:** For each row, perform the FSA API fetch and BigQuery update process.
- **Error Handling:** If a fetch fails for a specific coordinate pair:
    - Log the error details (timestamp, coordinates, error message).
    - **Continue** to the next coordinate pair in the list (Best Effort strategy).
    - The process should only report a global failure if *all* targets fail or a critical system error occurs.

## 3. Non-Functional Requirements
- **Performance:** Sequential processing of coordinates is acceptable (no strict need for parallelism yet).
- **Logging:** Distinct log entries for the start and end of each coordinate's processing cycle.

## 4. Out of Scope
- UI for managing the coordinate list (CRUD operations via Streamlit). Management will be done via BigQuery console or the setup script for now.
- Dynamic calculation of search radius (fixed or per-row config is fine).
