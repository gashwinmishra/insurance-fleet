output "cloud_run_service_url" {
  description = "The HTTPS URL of the deployed Agentic Insurance Fleet Cloud Run service."
  value       = google_cloud_run_v2_service.insurance_fleet.uri
}

output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository ID."
  value       = google_artifact_registry_repository.insurance_fleet.id
}

output "bigquery_table_fqn" {
  description = "Fully-qualified BigQuery table reference for claim logs."
  value       = "${var.project_id}.${google_bigquery_dataset.antigravity_logs.dataset_id}.${google_bigquery_table.claims_log.table_id}"
}

output "service_account_email" {
  description = "Least-privilege Service Account email used by Cloud Run."
  value       = google_service_account.fleet_runner.email
}
