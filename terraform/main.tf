terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com"
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Artifact Registry Repository for Container Images
resource "google_artifact_registry_repository" "insurance_fleet" {
  location      = var.region
  repository_id = "insurance-fleet"
  description   = "Docker container repository for Agentic Insurance Fleet"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# 3. BigQuery Dataset & Claims Telemetry Table
resource "google_bigquery_dataset" "antigravity_logs" {
  dataset_id                 = var.bigquery_dataset_id
  friendly_name              = "Antigravity Insurance Fleet Telemetry"
  description                = "Audit and Slow-Loop training dataset for multi-agent claim decisions"
  location                   = "US"
  delete_contents_on_destroy = false

  depends_on = [google_project_service.required_apis]
}

resource "google_bigquery_table" "claims_log" {
  dataset_id          = google_bigquery_dataset.antigravity_logs.dataset_id
  table_id            = var.bigquery_table_id
  deletion_protection = true

  schema = jsonencode([
    {
      name        = "timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "UTC timestamp of claim adjudication or human override"
    },
    {
      name        = "claim_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique UUID for the adjudicated claim"
    },
    {
      name        = "agent_decision"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "PII-redacted JSON payload containing Layer 2 outcome, trace_id, and latency_ms"
    },
    {
      name        = "human_correction"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "PII-redacted JSON payload of Human-in-the-Loop (HITL) adjuster override"
    },
    {
      name        = "environment"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Execution environment (LOCAL, STAGING, CLOUD)"
    }
  ])
}

# 4. Least-Privilege Service Account & IAM Bindings
resource "google_service_account" "fleet_runner" {
  account_id   = "insurance-fleet-sa"
  display_name = "Agentic Insurance Fleet Cloud Run Service Account"
  description  = "Least-privilege identity for Vertex AI Gemini inference and BigQuery telemetry"
}

resource "google_project_iam_member" "fleet_iam_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/logging.logWriter",
    "roles/secretmanager.secretAccessor"
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.fleet_runner.email}"
}

# 5. Secret Manager Configuration
resource "google_secret_manager_secret" "fleet_api_config" {
  secret_id = "insurance-fleet-config"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

# 6. Cloud Run v2 Managed Service
resource "google_cloud_run_v2_service" "insurance_fleet" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.fleet_runner.email

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      env {
        name  = "ENV"
        value = var.environment
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "SERVER_ADDRESS"
        value = "0.0.0.0"
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/_stcore/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_bigquery_table.claims_log
  ]
}
