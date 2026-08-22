## Consensus
All explorers agree on the strategy:
1. **`scripts/train_bqml_model.py`**: Add a `run_async: bool = False` argument to `train_model()`. If `run_async` is True, do not call `query_job.result()`; instead, return `query_job.job_id` so the job runs in the background on BigQuery.
2. **`app/ui/st_app.py`**: Import `train_model` from `scripts.train_bqml_model`. Add a button in the sidebar (e.g., under `🤖 ML Prediction Filters` or settings) that calls `train_model(..., run_async=True)` and displays the returned `job_id` in an `st.success` or `st.info` toast to provide immediate UI feedback without blocking.
3. **`tests/test_bqml_training.py`**: Create a test file that programmatically calls `train_model(..., dry_run=True)` to mathematically compile and validate the SQL query against the live BigQuery schema, satisfying the acceptance criteria.

## Resolved Conflicts
None. All explorers reached identical conclusions regarding the async mechanism, UI feedback, and dry-run validation.

## Action Plan for Worker
- Implement the async changes in `scripts/train_bqml_model.py`. Return the job ID when running asynchronously.
- Update `app/ui/st_app.py` with the UI button and success message in the sidebar. Note: the `project_id`, `dataset_id`, `table_id` can be parsed from `DEFAULT_BQ_PATH` which is usually defined in `st_app.py` or `bq_utils.py` (e.g. `DEFAULT_BQ_PATH.split('.')`).
- Create `tests/test_bqml_training.py` performing a dry-run test using the standard project, dataset, and table IDs from the codebase.
- Run `blaze build` or `pytest` to verify. (Note: use `pytest` as there is no blaze build in this python project).
