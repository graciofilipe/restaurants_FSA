# BQML Training Trigger UI Analysis

## Observation
1. **`scripts/train_bqml_model.py`**: The `train_model` function correctly implements the training query and already restricts the dataset using `WHERE user_rating IS NOT NULL`. It executes synchronously by calling `query_job.result()`, which would block the Streamlit UI. It also supports a `dry_run` mode via `bigquery.QueryJobConfig(dry_run=True)`. I successfully ran a local dry run via `python scripts/train_bqml_model.py --dry-run` which processed ~3.4MB and validated the query.
2. **`app/ui/st_app.py`**: The Streamlit application defines a sidebar for filters and actions. There is no existing section for model training. The app parses `DEFAULT_BQ_PATH` into `project_id`, `dataset_id`, and `table_id`, which are needed by `train_model`.
3. **`tests/`**: There is no existing test file thatprogrammatically runs the `--dry-run` pass for BQML training as requested in the acceptance criteria.

## Logic Chain
1. To implement asynchronous trigger logic (R2) without hanging the UI (R3), `scripts/train_bqml_model.py` needs a new parameter `run_async: bool = False` in the `train_model` function. If `True`, the function should skip `query_job.result()` and instead return `query_job.job_id`.
2. In `app/ui/st_app.py`, we need to import `train_model` from `scripts.train_bqml_model`.
3. In the sidebar section of `app/ui/st_app.py` (e.g., under `🤖 ML Prediction Filters`), we should add a new UI section and button:
   ```python
   st.divider()
   st.write("### 🧠 Model Training")
   if st.button("Train ML Model (Async)"):
       with st.spinner("Initiating training in BigQuery..."):
           from scripts.train_bqml_model import train_model
           try:
               job_id = train_model(
                   project_id, 
                   dataset_id, 
                   table_id, 
                   model_name="restaurant_preference_model", 
                   run_async=True
               )
               st.success(f"Training started! BigQuery Job ID: {job_id}")
               st.info("The UI is not blocked. Model training will complete in the background.")
           except Exception as e:
               st.error(f"Failed to start training: {e}")
   ```
4. To meet the acceptance criteria of a programmatic test, we must create `tests/test_bqml_training.py` that calls `train_model(..., dry_run=True)` to validate the SQL against the live BigQuery schema.

## Caveats
- The test `tests/test_bqml_training.py` will require valid Google Cloud Application Default Credentials to pass since `dry_run=True` interacts with the live BigQuery API.
- The UI will not poll BigQuery for the job completion status; it relies on BigQuery's fire-and-forget job execution.

## Conclusion
Modify `scripts/train_bqml_model.py` to support `run_async`, add the trigger button to the Streamlit sidebar in `app/ui/st_app.py`, and create `tests/test_bqml_training.py` to provide a programmatic dry-run validation test.

## Verification Method
- Execute `pytest tests/test_bqml_training.py` to verify the `--dry-run` test compiles mathematically without errors.
- Run `streamlit run app/ui/st_app.py`, click "Train ML Model (Async)" in the sidebar, and confirm that the UI immediately returns a success message containing the Job ID without freezing the browser.
