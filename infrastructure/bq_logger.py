import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from infrastructure.observability import PIIRedactor


class BQLogger:
    """
    Enterprise BigQuery Audit & Telemetry Sink with Automatic PII Redaction & Async Support.
    Scrubs PII via `PIIRedactor` before streaming structured agent decisions, execution
    trace IDs, and human-in-the-loop corrections to BigQuery (`antigravity_logs.claims_log`).
    Provides both synchronous (`log_feedback`) and non-blocking asynchronous (`alog_feedback`) APIs.
    """

    _recent_logs: List[Dict[str, Any]] = []
    _async_lock: Optional[asyncio.Lock] = None

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

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._async_lock is None:
            cls._async_lock = asyncio.Lock()
        return cls._async_lock

    def _build_redacted_entry(
        self,
        claim_id: str,
        agent_decision: Any,
        human_correction: Optional[Any] = None,
        trace_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        raw_decision = (
            dict(agent_decision) if isinstance(agent_decision, dict) else {"value": agent_decision}
        )
        if trace_id:
            raw_decision["trace_id"] = trace_id
        if latency_ms is not None:
            raw_decision["latency_ms"] = round(float(latency_ms), 2)

        sanitized_decision, dec_redactions = PIIRedactor.redact(raw_decision)
        sanitized_correction, corr_redactions = (
            PIIRedactor.redact(human_correction) if human_correction is not None else (None, 0)
        )
        sanitized_decision["pii_redactions_applied"] = dec_redactions + corr_redactions

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "claim_id": claim_id,
            "agent_decision": json.dumps(sanitized_decision),
            "human_correction": (
                json.dumps(sanitized_correction)
                if isinstance(sanitized_correction, (dict, list))
                else sanitized_correction
            ),
            "environment": self.env,
        }

    def log_feedback(
        self,
        claim_id: str,
        agent_decision: Any,
        human_correction: Optional[Any] = None,
        trace_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Synchronous audit logger with automatic PII redaction prior to storage."""
        log_entry = self._build_redacted_entry(
            claim_id=claim_id,
            agent_decision=agent_decision,
            human_correction=human_correction,
            trace_id=trace_id,
            latency_ms=latency_ms,
        )

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
            print(f"\033[92m[BQ_LOCAL] Logged (PII-Sanitized): {json.dumps(log_entry, indent=2)}\033[0m")

        return log_entry

    async def alog_feedback(
        self,
        claim_id: str,
        agent_decision: Any,
        human_correction: Optional[Any] = None,
        trace_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Non-blocking asynchronous BigQuery & local audit logging to prevent UI blocking."""
        async with self._get_lock():
            return await asyncio.to_thread(
                self.log_feedback,
                claim_id,
                agent_decision,
                human_correction,
                trace_id,
                latency_ms,
            )

    @classmethod
    def get_recent_logs(cls, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(cls._recent_logs[-limit:]))
