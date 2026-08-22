# Handoff: BQML Async Trigger Feature

## Observation
1. `scripts/train_bqml_model.py` defines `train_model()`, which currently executes the BQML query synchronously (blocks on `query_job.result()`) unless `dry_run=True`. It already correctly filters the data with `WHERE user_rating IS NOT NULL`.
2. `app/ui/st_app.py` contains the Streamlit sidebar (lines ~188-256) where ML-related configuration and filters reside.
3. The acceptance criteria demands that model training can be triggered from the Streamlit UI without causing application hang (Async mode), returning user feedback (Job ID), and that a dry-run script/test validates the SQL. `train_bqml_model.py` already implements `--dry-run` using BigQuery's `dry_run=True` config, satisfying the last requirement natively.

## Logic Chain
1. To prevent UI hang, `train_model` must support an asynchronous mode where it fires the BigQuery job and returns the Job ID immediately instead of calling `.result()`.
2. Modifying `train_bqml_model.py` to accept a `sync: bool = True` parameter will allow backwards compatibility for existing synchronous runs while giving the UI the ability to pass `sync=False`.
3. In `st_app.py`, we can import `train_model` from `scripts.train_bqml_model` (as the app is executed from the project root) and add a sidebar button ("Trigger BQML Training") under the "ML Prediction Filters" section.
4. Upon clicking the button, Streamlit calls `train_model(sync=False)` and uses the returned BigQuery job ID to show a `st.success` message to the user, fulfilling the feedback requirement.
5. The dry-run validation requirement is already fulfilled by running `python scripts/train_bqml_model.py --dry-run` from the command line, which uses the BigQuery dry run functionality to mathematically compile the model creation statement against the live schema. 

## Caveats
- Since we are calling BigQuery, the Streamlit app's environment needs GCP Application Default Credentials, which is expected based on project context.
- There is no automated polling implemented in Streamlit for the background training job, as it's a "fire-and-forget" action. The user gets the job ID and can check BigQuery for completion.
- We assume Python 3 namespace packages will correctly resolve `from scripts.train_bqml_model import train_model` from within `st_app.py`, otherwise `sys.path.append(...)` might be required.

## Conclusion
The implementation should proceed by:
1. Modifying `scripts/train_bqml_model.py` to add `sync=True` to `train_model`, returning the `query_job` (or `job_id`) immediately if `sync=False`.
2. Updating `app/ui/st_app.py` to include a button in the sidebar (under the ML section) that calls `train_model(..., sync=False)`.
3. Displaying the resulting Job ID via `st.success()` in Streamlit.
4. Ensuring no other blocking calls exist in the trigger path.

## Verification Method
1. **Dry-run validation**: Execute `python3 scripts/train_bqml_model.py --dry-run`. It should output "Dry run successful. Query is valid."
2. **UI Integration**: Start Streamlit with `streamlit run app/ui/st_app.py`. Click the new "Trigger BQML Training" button in the sidebar. The UI should instantly display a success message with a Job ID, and the application should remain responsive.
3. **Backend check**: Verify in the Google Cloud Console (BigQuery) that a training job for `restaurant_preference_model` was launched.
