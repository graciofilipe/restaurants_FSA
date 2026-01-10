# Plan: Scheduled Discovery & Config Update

## Phase 1: Update Search Configuration
- [ ] Task: Create a python script `scripts/update_search_config.py` to manage the `config_search_params` table (add/remove/update rows).
- [ ] Task: Ask user for new coordinates and update the BigQuery table using the script.
- [ ] Task: Verify the table content reflects the desired configuration.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Update Search Configuration' (Protocol in workflow.md)

## Phase 2: Cloud Job Deployment
- [ ] Task: Update `scripts/create_cloud_run_job.sh` to ensure it uses the correct Service Account and permissions if needed.
- [ ] Task: Execute `scripts/create_cloud_run_job.sh` to create/update the Cloud Run Job `fetch-weekly`.
- [ ] Task: Create a Cloud Scheduler job `trigger-fetch-weekly` to invoke the Cloud Run Job (e.g., weekly schedule).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Cloud Job Deployment' (Protocol in workflow.md)

## Phase 3: Final Verification
- [ ] Task: Manually trigger the Cloud Run Job via CLI (`gcloud run jobs execute`).
- [ ] Task: Verify execution logs and BigQuery for new data ingestion.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Final Verification' (Protocol in workflow.md)
