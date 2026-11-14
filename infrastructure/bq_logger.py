import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class BQLogger:
    """
    Enterprise BigQuery Audit & Telemetry Sink.
    Streams structured agent decisions, execution trace IDs, and human-in-the-loop
    corrections to BigQuery (`antigravity_logs.claims_log`) in CLOUD mode, while
    maintaining a local structured ring buffer for real-time observability inspection.
    """

    _recent_logs: List[Dict[str, Any]] = []

    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "antigravity_logs",
        table_id: str = "claims_log",
    ):
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.env = os.getenv("ENV", "LOCAL")

    def log_feedback(
        self,
        claim_id: str,
        agent_decision: Any,
        human_correction: Optional[Any] = None,
        trace_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Logs agent decisions and optional human corrections to BigQuery (or local audit sink).
        Preserves the 5-column BigQuery table schema (`timestamp`, `claim_id`, `agent_decision`,
        `human_correction`, `environment`) by embedding trace telemetry inside `agent_decision`.
        """
        decision_payload = (
            dict(agent_decision) if isinstance(agent_decision, dict) else {"value": agent_decision}
        )
        if trace_id:
            decision_payload["trace_id"] = trace_id
        if latency_ms is not None:
            decision_payload["latency_ms"] = round(float(latency_ms), 2)

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_id": claim_id,
            "agent_decision": json.dumps(decision_payload),
            "human_correction": (
                json.dumps(human_correction)
                if isinstance(human_correction, (dict, list))
                else human_correction
            ),
            "environment": self.env,
        }

        BQLogger._recent_logs.append(log_entry)
        if len(BQLogger._recent_logs) > 100:
            BQLogger._recent_logs = BQLogger._recent_logs[-100:]

        if self.env == "CLOUD":
            try:
                from google.cloud import bigquery

                client = bigquery.Client(project=self.project_id)
                table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
                errors = client.insert_rows_json(table_ref, [log_entry])

                if errors:
                    print(f"[BQ_ERROR] Encountered errors while inserting rows: {errors}")
                else:
                    print(f"[BQ_CLOUD] Successfully logged claim {claim_id}")

            except ImportError:
                print("[BQ_ERROR] google-cloud-bigquery library not found.")
            except Exception as e:
                print(f"[BQ_ERROR] Failed to insert log: {e}")
        else:
            print(f"\033[92m[BQ_LOCAL] Logged: {json.dumps(log_entry, indent=2)}\033[0m")

        return log_entry

    @classmethod
    def get_recent_logs(cls, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(cls._recent_logs[-limit:]))

