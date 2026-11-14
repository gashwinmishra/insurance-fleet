import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple


class PIIRedactor:
    """
    Enterprise PII & Sensitive Data Redaction Engine (Pillar 4: Observability & Tracing).
    Automatically scrubs and masks Personally Identifiable Information (PII), financial
    identifiers, vehicle identifiers (VIN/Plates), and credentials from logs, spans,
    tool inputs/outputs, and BigQuery telemetry payloads before persistence.
    """

    SENSITIVE_KEYS = {
        "ssn",
        "social_security",
        "tax_id",
        "email",
        "phone",
        "phone_number",
        "credit_card",
        "card_number",
        "cvv",
        "vin",
        "license_plate",
        "policyholder_name",
        "customer_name",
        "home_address",
        "street_address",
        "api_key",
        "secret",
        "password",
        "token",
        "authorization",
    }

    PATTERNS: List[Tuple[re.Pattern, str]] = [
        # Social Security Numbers (XXX-XX-XXXX)
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
        # Email Addresses
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        # Credit Card Numbers (13-16 digits)
        (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CC]"),
        # US/International Phone Numbers
        (re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), "[REDACTED_PHONE]"),
        # 17-character standard Vehicle Identification Numbers (VIN)
        (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b"), "[REDACTED_VIN]"),
        # API Keys & Bearer Tokens
        (re.compile(r"(?:Bearer\s+[A-Za-z0-9\-._~+/]+=*|AIza[0-9A-Za-z\-_]{35}|gh[pousr]_[A-Za-z0-9_]{20,})"), "[REDACTED_SECRET]"),
    ]

    @classmethod
    def redact_text(cls, text: str) -> Tuple[str, int]:
        """Scrubs inline PII patterns from a string and returns (redacted_text, redaction_count)."""
        if not text or not isinstance(text, str):
            return str(text or ""), 0
        count = 0
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized, num_subs = pattern.subn(replacement, sanitized)
            count += num_subs
        return sanitized, count

    @classmethod
    def redact(cls, data: Any) -> Tuple[Any, int]:
        """
        Recursively sanitizes strings, dicts, and lists, masking both sensitive keys
        and inline regex matches. Returns (sanitized_data, total_redaction_count).
        """
        if isinstance(data, str):
            return cls.redact_text(data)

        if isinstance(data, dict):
            total_count = 0
            sanitized_dict: Dict[str, Any] = {}
            for k, v in data.items():
                key_lower = str(k).lower().strip()
                if key_lower in cls.SENSITIVE_KEYS and v is not None:
                    sanitized_dict[k] = f"[REDACTED_{key_lower.upper()}]"
                    total_count += 1
                else:
                    clean_v, c = cls.redact(v)
                    sanitized_dict[k] = clean_v
                    total_count += c
            return sanitized_dict, total_count

        if isinstance(data, list):
            total_count = 0
            sanitized_list = []
            for item in data:
                clean_item, c = cls.redact(item)
                sanitized_list.append(clean_item)
                total_count += c
            return sanitized_list, total_count

        return data, 0


class StructuredJsonFormatter(logging.Formatter):
    """Formats standard Python log records as Cloud Logging compatible JSON with PII redaction."""

    def format(self, record: logging.LogRecord) -> str:
        clean_msg, msg_redactions = PIIRedactor.redact_text(record.getMessage())
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": clean_msg,
            "pii_redactions": msg_redactions,
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = getattr(record, "trace_id")
        if hasattr(record, "span_id"):
            payload["span_id"] = getattr(record, "span_id")
        if hasattr(record, "extra_fields") and isinstance(getattr(record, "extra_fields"), dict):
            clean_extra, extra_redactions = PIIRedactor.redact(getattr(record, "extra_fields"))
            payload.update(clean_extra)
            payload["pii_redactions"] += extra_redactions
        return json.dumps(payload)


def get_structured_logger(name: str = "insurance_fleet") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


class Span:
    """Represents a single timed operation or agent/tool execution span with automatic PII scrubbing."""

    def __init__(
        self,
        name: str,
        agent: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.span_id: str = f"span_{uuid.uuid4().hex[:12]}"
        self.trace_id: str = trace_id
        self.parent_span_id: Optional[str] = parent_span_id
        self.name: str = name
        self.agent: str = agent
        self.start_time: float = time.perf_counter()
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None
        self.duration_ms: float = 0.0
        self.status: str = "IN_PROGRESS"
        clean_attrs, redaction_cnt = PIIRedactor.redact(dict(attributes or {}))
        self.attributes: Dict[str, Any] = clean_attrs
        self.pii_redactions_count: int = redaction_cnt
        self.tool_calls: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_output: Any,
        duration_ms: float = 0.0,
    ) -> None:
        clean_args, args_redacted = PIIRedactor.redact(tool_args)
        clean_output, out_redacted = PIIRedactor.redact_text(str(tool_output)[:400])
        self.pii_redactions_count += args_redacted + out_redacted
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": clean_args,
            "output_preview": clean_output,
            "duration_ms": round(duration_ms, 2),
            "pii_redacted": (args_redacted + out_redacted) > 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        clean_meta, meta_redacted = PIIRedactor.redact(metadata or {})
        self.pii_redactions_count += meta_redacted
        self.events.append({
            "event": event_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": clean_meta,
        })

    def finish(self, status: str = "OK", attributes: Optional[Dict[str, Any]] = None) -> None:
        self.duration_ms = round((time.perf_counter() - self.start_time) * 1000.0, 2)
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status
        if attributes:
            clean_attrs, c = PIIRedactor.redact(attributes)
            self.attributes.update(clean_attrs)
            self.pii_redactions_count += c

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "agent": self.agent,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "pii_redactions_count": self.pii_redactions_count,
            "attributes": self.attributes,
            "tool_calls": self.tool_calls,
            "events": self.events,
        }


class ObservabilityTracer:
    """
    End-to-End Multi-Agent Distributed Tracer with Automatic PII Redaction.
    Captures spans, tool invocations (intent vs. outcome), model routing telemetry,
    PII redaction counts, confidence drift across layers, and latency waterfalls.
    """

    def __init__(self, claim_id: Optional[str] = None):
        self.trace_id: str = f"trace_{uuid.uuid4().hex[:16]}"
        self.claim_id: str = claim_id or f"claim_{uuid.uuid4().hex[:8]}"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.start_perf: float = time.perf_counter()
        self.spans: List[Span] = []
        self._active_span: Optional[Span] = None
        self.logger = get_structured_logger("insurance_fleet.tracer")

    @contextmanager
    def start_span(
        self,
        name: str,
        agent: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        parent_id = self._active_span.span_id if self._active_span else None
        span = Span(
            name=name,
            agent=agent,
            trace_id=self.trace_id,
            parent_span_id=parent_id,
            attributes=attributes,
        )
        prev_span = self._active_span
        self._active_span = span
        self.spans.append(span)
        try:
            yield span
            if span.status == "IN_PROGRESS":
                span.finish(status="OK")
        except Exception as exc:
            span.finish(status="ERROR", attributes={"error": str(exc)})
            raise
        finally:
            self._active_span = prev_span
            self.logger.info(
                f"Span completed: {agent}.{name} ({span.duration_ms}ms) [{span.status}]",
                extra={
                    "trace_id": self.trace_id,
                    "span_id": span.span_id,
                    "extra_fields": {
                        "agent": agent,
                        "operation": name,
                        "duration_ms": span.duration_ms,
                        "status": span.status,
                        "pii_redactions_count": span.pii_redactions_count,
                    },
                },
            )

    def build_summary(self, decision_trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_duration_ms = round((time.perf_counter() - self.start_perf) * 1000.0, 2)
        tool_call_count = sum(len(s.tool_calls) for s in self.spans)
        total_pii_redactions = sum(s.pii_redactions_count for s in self.spans)

        confidence_progression: Dict[str, float] = {}
        outcome_progression: Dict[str, str] = {}
        if isinstance(decision_trace, dict):
            l1 = decision_trace.get("layer_1_base", {})
            l2 = decision_trace.get("layer_2_inner", {}).get("decision", {})
            l3 = decision_trace.get("layer_3_outer", {}).get("decision", {})
            confidence_progression = {
                "layer_1_base": float(l1.get("confidence", 0.0)),
                "layer_2_inner": float(l2.get("confidence", l1.get("confidence", 0.0))),
                "layer_3_outer": float(l3.get("confidence", 0.0)),
            }
            outcome_progression = {
                "layer_1_base": str(l1.get("outcome", "UNKNOWN")),
                "layer_2_inner": str(l2.get("outcome", "UNKNOWN")),
                "layer_3_outer": str(l3.get("outcome", "UNKNOWN")),
            }

        return {
            "trace_id": self.trace_id,
            "claim_id": self.claim_id,
            "created_at": self.created_at,
            "total_duration_ms": total_duration_ms,
            "span_count": len(self.spans),
            "tool_call_count": tool_call_count,
            "pii_redactions_applied": total_pii_redactions,
            "confidence_progression": confidence_progression,
            "outcome_progression": outcome_progression,
            "spans": [s.to_dict() for s in self.spans],
        }
