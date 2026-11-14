# 🚀 Deployment Guide - Agentic Insurance Fleet

This guide will help you deploy the "Antigravity Insurance" application to Google Cloud Platform (GCP).

## 1. Prerequisites
Ensure you have the Google Cloud SDK installed and authenticated:
```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

## 2. Infrastructure Setup (BigQuery)
Create the BigQuery dataset and table for logging decisions.

```bash
# Create Dataset
bq --location=US mk -d antigravity_logs

# Create Table (using a schema definition)
bq mk --table antigravity_logs.claims_log \
  timestamp:TIMESTAMP,claim_id:STRING,agent_decision:JSON,human_correction:JSON,environment:STRING
```

## 3. Permissions Setup (Critical)
Cloud Build needs permission to push images and write logs. Run these commands ensuring you replace `[YOUR_PROJECT_ID]` with your actual project ID (e.g., `ashwin-genai-sandbox`):

```bash
# Set your project ID variable
export PROJECT_ID=[YOUR_PROJECT_ID]
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# Grant Artifact Registry Repo Admin (needed to CREATE the repo on first push)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
    --role=roles/artifactregistry.repoAdmin

# Grant Logging Writer (to see build logs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
    --role=roles/logging.logWriter
    
# Grant Storage Admin (often needed for legacy GCR or caching)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
    --role=roles/storage.admin
```

## 4. One-Time Setup: Create Artifact Repository
Since we are moving away from legacy `gcr.io`, we must create a Docker repository in Artifact Registry first.

```bash
gcloud artifacts repositories create insurance-fleet \
    --repository-format=docker \
    --location=us-central1 \
    --description="Repository for Insurance Fleet App"
```

## 5. Deployment (Cloud Run)
Now you can deploy using Cloud Build. It will define the image in `us-central1-docker.pkg.dev`.

```bash
# Submit the build
gcloud builds submit --config cloudbuild.yaml .
```

This command will:
1.  **Build** the Docker image.
2.  **Push** it to Google Container Registry (GCR) or Artifact Registry.
3.  **Deploy** a new Cloud Run service named `insurance-fleet-demo`.

## 4. Environment Variables
The `cloudbuild.yaml` automatically sets:
*   `ENV=CLOUD` (Triggers real BigQuery logging)
*   `PROJECT_ID` (Your GCP Project ID)

## 5. View Your App
Once deployment finishes, `gcloud` will output a Service URL (e.g., `https://insurance-fleet-demo-xyz-uc.a.run.app`).
Click strictly on that link to see your live production app!

## 6. Accessing the App (Secure Proxy)
**NOTE:** Your Organization Policy prevents public access (`allUsers`).

To access the deployed app securely from your local machine, run this command:
```bash
gcloud run services proxy insurance-fleet-demo --region=us-central1 --port=8080
```

Then open your browser to: **http://localhost:8080**
(The app is running on Cloud Run, but accessed via a secure tunnel to your machine).

## 7. Verify Logging
After processing a few claims in the live app, query BigQuery to see the persistent logs:
```sql
SELECT * FROM `[YOUR_PROJECT_ID].antigravity_logs.claims_log` ORDER BY timestamp DESC
```
