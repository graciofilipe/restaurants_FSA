## Review Summary

**Verdict**: REQUEST_CHANGES

## Findings

### [Critical] Finding 1: Denial-of-Wallet via Unrestricted Concurrent Training
- What: The "Train BQML Model (Async)" button allows spawning unlimited concurrent background BQML training jobs.
- Where: `app/ui/st_app.py`, line 233 (`if st.button("Train BQML Model (Async)"):`)
- Why: BigQuery ML training is compute-intensive and costly. A user clicking the button multiple times, accidentally or intentionally, will queue multiple `CREATE OR REPLACE MODEL` jobs.
- Suggestion: Disable the button while a job is running (using `st.session_state`) or implement a debounce/check for active jobs in BQ before allowing a new run.

### [Major] Finding 2: Uncommitted Test Files
- What: Several test files related to the async functionality were created but not committed.
- Where: Working directory (`tests/test_async_bqml.py`, `tests/test_bqml_training_stress.py`).
- Why: CI/CD won't run these tests because they are untracked.
- Suggestion: Run `git add` for the new test files and commit them.

### [Major] Finding 3: UI Ignores User's BQ Path Input
- What: The `bq_path_input` text input is instantiated but never used. The app continues to use the hardcoded `bq_path` variable.
- Where: `app/ui/st_app.py`, line 190. 
- Why: If a user tries to train the model on a different dataset by modifying the path in the sidebar, the script still uses `DEFAULT_BQ_PATH`.
- Suggestion: Parse `bq_path_input` to update `project_id`, `dataset_id`, and `table_id` before passing them to `train_model`.

### [Minor] Finding 4: No Visibility into Async Job Status
- What: The UI returns a job ID in a transient success toast (`st.success`), but there is no persistent UI to check if the background job succeeded or failed.
- Where: `app/ui/st_app.py`
- Why: If the training fails asynchronously due to missing data or schema issues, the user will never be notified.
- Suggestion: Store the latest `job_id` in `st.session_state` and display its status, or provide a "Check Job Status" button.

## Verified Claims
- `scripts/train_bqml_model.py` correctly implements `run_async` returning the `job_id` → verified via code inspection → PASS
- Async and sync testing coverage exists → verified via `pytest` run of untracked files → PASS (but files are uncommitted)
- Dry run test validates against BQ → verified via `pytest tests/test_bqml_training.py` → PASS

## Challenge Summary

**Overall risk assessment**: HIGH

## Challenges

### [High] Challenge 1: Resource Exhaustion (DoW)
- Assumption challenged: Users will only click the button once and wait.
- Attack scenario: User clicks "Train" 10 times in 5 seconds.
- Blast radius: 10 concurrent BQML training jobs are launched. This wastes BigQuery slot capacity, incurs high costs, and could lead to quota exhaustion (`quotaExceeded` for concurrent jobs).
- Mitigation: Disable the button using `st.session_state` locks and add a visual indicator that training is ongoing.

### [Medium] Challenge 2: Silent Async Failures
- Assumption challenged: The job will successfully complete in the background if `client.query()` doesn't immediately raise an error.
- Attack scenario: The dataset is large enough that BQ accepts the job, but during execution, it hits a data skew error or timeout.
- Blast radius: The user thinks the model is trained but it is not.
- Mitigation: Provide a mechanism to poll the job ID or alert the user on failure.
