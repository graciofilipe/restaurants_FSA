#!/bin/bash
# Helper script to create/update the Cloud Run Job for Weekly Fetch
# Usage: ./scripts/create_cloud_run_job.sh [IMAGE_URL]

IMAGE_URL=${1:-"gcr.io/${GOOGLE_CLOUD_PROJECT}/python-app:latest"}
REGION="europe-west2"
JOB_NAME="fetch-weekly"

echo "Creating/Updating Cloud Run Job: $JOB_NAME using image: $IMAGE_URL"

gcloud run jobs create $JOB_NAME \
    --image $IMAGE_URL \
    --command "python" \
    --args "-m,app.cron.fetch_weekly" \
    --region $REGION \
    --task-timeout 3600s \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT \
    || gcloud run jobs update $JOB_NAME \
        --image $IMAGE_URL \
        --command "python" \
        --args "-m,app.cron.fetch_weekly" \
        --region $REGION \
        --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT
