# Original User Request

## Initial Request — 2026-06-08T16:18:07Z

Build a feature in the Streamlit application that allows the user to trigger the BigQuery ML (`train_bqml_model`) training pipeline directly from the UI. The training process should automatically use all restaurants that have an assigned `user_rating` as the training data.

Working directory: /usr/local/google/home/filipegracio/restaurants/restaurants_FSA

## Requirements

### R1. UI Integration
Add a dedicated button or section in the Streamlit app (`app/ui/st_app.py`)—such as in the sidebar or a settings tab—that allows the user to explicitly trigger a model retraining.

### R2. Asynchronous Trigger Logic
Connect the UI button to the existing model training logic found in `scripts/train_bqml_model.py`. Because the `BOOSTED_TREE_REGRESSOR` takes 10-15 minutes to train, the trigger must implement a "fire and forget" asynchronous pattern. It should start the BigQuery job in the background and immediately return control to the UI.

### R3. User Feedback
The UI must provide immediate, clear feedback to the user (e.g., a toast or success message) confirming that the model training job has successfully started in the background.

## Acceptance Criteria

### Workflow Verification
- [ ] The model training can be triggered from the Streamlit UI without causing the browser or application to hang.
- [ ] A programmatic script or test exists that executes a real `--dry-run` pass of the training pipeline locally, verifying that the `CREATE OR REPLACE MODEL` SQL command mathematically compiles and validates against the live BigQuery schema without errors.
