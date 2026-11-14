variable "project_id" {
  description = "The Google Cloud Project ID hosting the Agentic Insurance Fleet."
  type        = string
}

variable "region" {
  description = "Primary GCP region for Cloud Run, Artifact Registry, and Vertex AI."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment identifier (e.g., CLOUD, STAGING, PROD)."
  type        = string
  default     = "CLOUD"
}

variable "service_name" {
  description = "Name of the Cloud Run v2 service."
  type        = string
  default     = "insurance-fleet-demo"
}

variable "bigquery_dataset_id" {
  description = "BigQuery dataset ID for storing agent decisions and HITL overrides."
  type        = string
  default     = "antigravity_logs"
}

variable "bigquery_table_id" {
  description = "BigQuery table ID for claim adjudication telemetry."
  type        = string
  default     = "claims_log"
}

variable "container_image" {
  description = "Full Artifact Registry container image URI to deploy."
  type        = string
  default     = "us-central1-docker.pkg.dev/placeholder-project/insurance-fleet/app:latest"
}
