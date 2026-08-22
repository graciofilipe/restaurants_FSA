# Handoff Report: BQML Async Trigger UI Implementation

## Observation
1. **Current Synchronous Behavior:** `scripts/train_bqml_model.py`'s `train_model()` function currently executes the BQML training query synchronously by calling `query_job.result()` (line 70), which blocks the Python process until the 10-15 minute job completes.
2. **Built-in Dry Run Support:** `scripts/train_bqml_model.py` already handles `--dry-run` validation perfectly using `bigquery.QueryJobConfig(dry_run=True)` (lines 58-62).
3. **Training Data Requirements:** The SQL query in `train_model()` inherently isolates training data to user-rated restaurants via the pre-existing `WHERE user_rating IS NOT NULL` clause (line 52).
4. **Streamlit UI Extensibility:** `app/ui/st_app.py` has a dedicated `with st.sidebar:` context where application-wide configuration and actions reside, making it an ideal location for the trigger button. The `scripts` directory has an `__init__.py` file, allowing direct importation of `train_model` into Streamlit.
5. **Testing Architecture:** The codebase uses `pytest`, and `cloudbuild.yaml` automatically executes tests. There is an existing `tests/` directory ready for a programmatic dry-run test.

## Logic Chain
1. **Asynchronous Execution (R2):** Since the BigQuery Python client's `query()` method initiates jobs asynchronously on Google's infrastructure, we can achieve fire-and-forget execution by adding a `run_async: bool = False` parameter to `train_model()`. If true, the function returns `query_job.job_id` *without* invoking `.result()`. This offloads the wait to BigQuery, satisfying the non-blocking requirement.
2. **UI Implementation & User Feedback (R1 & R3):** We will import `train_model` into `app/ui/st_app.py` and inject a "Trigger Training Pipeline" button in the sidebar. Upon click, it will call `train_model(..., run_async=True)` and immediately display the returned `job_id` via `st.success()`.
3. **Acceptance Criteria for Dry Run:** Creating `tests/test_bqml_dry_run.py` to invoke `train_model(..., dry_run=True)` fulfills the requirement. A dry run execution asks the BigQuery engine to mathematically compile the `CREATE OR REPLACE MODEL` SQL and validate it against the live schema without consuming compute resources. If the test passes without raising a `GoogleCloudError`, the query is definitively structurally valid.

## Caveats
- **Status Polling:** The Streamlit UI will implement a "fire-and-forget" trigger. It will successfully show that the job has started, but it will not continuously poll BigQuery to notify the user when the 10-15 minute job is complete. The user would need to check the ML Predictions metrics to see if the model has been refreshed.
- **Test Authentication:** The real `--dry-run` test operates against the live BigQuery instance. It requires active Google Cloud ADC (Application Default Credentials). If the CI environment lacks BQ permissions, the test will fail in the pipeline.

## Conclusion
The implementation is straightforward and requires three key steps:
1. **Modify `scripts/train_bqml_model.py`**: Update `train_model()` to accept `run_async`. If `run_async` is True, bypass `query_job.result()` and return `query_job.job_id`.
2. **Update `app/ui/st_app.py`**: Add `from scripts.train_bqml_model import train_model`. In the sidebar, add a button that invokes the function asynchronously and outputs user feedback using `st.success` and `st.info`.
3. **Add `tests/test_bqml_dry_run.py`**: Create a programmatic test that calls `train_model(..., dry_run=True)` using the production project ID and dataset, verifying no exceptions are raised.

## Verification Method
1. **Dry-Run Validation:** Run `pytest tests/test_bqml_dry_run.py`. It should execute instantly and pass, confirming BigQuery accepts the schema and logic.
2. **UI Trigger Validation:** Execute `streamlit run app/ui/st_app.py`. Click the new "Trigger Training Pipeline" button in the sidebar. The UI should immediately return a success message containing the Job ID without freezing the browser or hanging the app.
3. **Background Job Validation:** Open the Google Cloud BigQuery Console and verify that a model training job matching the output Job ID was successfully submitted and is running.
