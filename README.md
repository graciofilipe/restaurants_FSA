# restaurants_FSA
uses public FSA data to extract restaurant information

## Streamlit Application

This repository contains a simple Streamlit application.

## Running Locally with Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t streamlit-app .
    ```
2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 streamlit-app
    ```
    The application will be accessible at `http://localhost:8501`.

## Deploying to Google Cloud Run

This project can be deployed to Google Cloud Run using a Cloud Build trigger.

1.  **Prerequisites:**
    *   A Google Cloud Project with billing enabled.
    *   Cloud Build API and Cloud Run API enabled in your GCP project.
    *   Your repository (e.g., on GitHub, Bitbucket, or Cloud Source Repositories) connected to Google Cloud Build.

2.  **Configure a Cloud Build Trigger:**
    *   In the Google Cloud Console, navigate to Cloud Build and create a new trigger.
    *   Connect it to your source code repository.
    *   Configure the trigger to build from your `Dockerfile` when changes are pushed to your desired branch (e.g., `main` or `master`).
    *   In the build steps for the trigger, ensure it performs the following actions:
        1.  Builds the Docker image using the `Dockerfile`:
            `docker build -t gcr.io/$PROJECT_ID/$REPO_NAME:$COMMIT_SHA .`
        2.  Pushes the image to Google Container Registry (GCR):
            `docker push gcr.io/$PROJECT_ID/$REPO_NAME:$COMMIT_SHA`
        3.  Deploys the image to Cloud Run:
            `gcloud run deploy $REPO_NAME --image gcr.io/$PROJECT_ID/$REPO_NAME:$COMMIT_SHA --platform managed --region YOUR_REGION --allow-unauthenticated`
            (Replace `YOUR_REGION` with your desired Google Cloud region, e.g., `us-central1`. You can also parameterize `$REPO_NAME` or hardcode your service name).

3.  **Triggering a Deployment:**
    Pushing changes to the configured branch in your repository will automatically trigger the Cloud Build pipeline, which will build and deploy your Streamlit application to Cloud Run.

4.  **Access the application:**
    Once deployed, Google Cloud Run will provide a URL to access your application.
