# 🚀 Deployment Guide - Agentic Insurance Fleet

This guide explains how to provision and deploy the **Agentic Insurance Fleet** on Google Cloud Platform (GCP) using declarative **Terraform Infrastructure as Code (IaC)** and **Cloud Build CI/CD**.

## 1. Declarative Infrastructure Provisioning (Terraform — Recommended)

All GCP resources — including **Artifact Registry**, **BigQuery** (`antigravity_logs.claims_log`), **Least-Privilege Service Account & IAM**, **Secret Manager**, and **Cloud Run v2** — are declaratively defined in [`terraform/`](./terraform/main.tf).

```bash
cd terraform

# Initialize Terraform providers
terraform init

# Preview declarative execution plan
terraform plan -var="project_id=YOUR_PROJECT_ID"

# Apply infrastructure changes
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### Resources Provisioned by Terraform:
- `google_artifact_registry_repository.insurance_fleet`: Docker repository in `us-central1`.
- `google_bigquery_dataset.antigravity_logs` & `google_bigquery_table.claims_log`: Typed JSON telemetry table for PII-redacted agent decisions and HITL corrections.
- `google_service_account.fleet_runner`: Dedicated runtime service account with least-privilege roles (`roles/aiplatform.user`, `roles/bigquery.dataEditor`, `roles/logging.logWriter`, `roles/secretmanager.secretAccessor`).
- `google_secret_manager_secret.fleet_api_config`: Managed secret store.
- `google_cloud_run_v2_service.insurance_fleet`: Managed Cloud Run v2 service with `/_stcore/health` startup probes and autoscaling.

---

## 2. CI/CD Automated Evaluation & Container Deployment (Cloud Build)

The [`cloudbuild.yaml`](./cloudbuild.yaml) pipeline automatically runs:
1. **5-Pillar Unit & Integration Test Suite** (`tests/test_fleet.py`)
2. **Generative Evaluation Suite against the Golden Dataset** (`evals/run_eval.py` against `evals/golden_dataset.json`)
3. **Docker Image Build & Artifact Registry Push**
4. **Cloud Run Deployment**

```bash
gcloud builds submit --config cloudbuild.yaml .
```

---

## 3. Verify BigQuery Telemetry

After processing claims, query BigQuery to inspect the PII-redacted decision logs:
```sql
SELECT timestamp, claim_id, agent_decision, human_correction, environment
FROM `YOUR_PROJECT_ID.antigravity_logs.claims_log`
ORDER BY timestamp DESC
LIMIT 20;
```
