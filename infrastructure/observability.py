import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional


class StructuredJsonFormatter(logging.Formatter):
    """Formats standard Python log records as Cloud Logging compatible JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "trace_id"):
            payload["trace_id"] = getattr(record, "trace_id")
        if hasattr(record, "span_id"):
            payload["span_id"] = getattr(record, "span_id")
        if hasattr(record, "extra_fields") and isinstance(getattr(record, "extra_fields"), dict):
            payload.update(getattr(record, "extra_fields"))
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
    """Represents a single timed operation or agent/tool execution span."""

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
        self.attributes: Dict[str, Any] = dict(attributes or {})
        self.tool_calls: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_output: Any,
        duration_ms: float = 0.0,
    ) -> None:
        self.tool_calls.append({
            "tool_name": tool_name,
            "arguments": tool_args,
            "output_preview": str(tool_output)[:300],
            "duration_ms": round(duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "event": event_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })

    def finish(self, status: str = "OK", attributes: Optional[Dict[str, Any]] = None) -> None:
        self.duration_ms = round((time.perf_counter() - self.start_time) * 1000.0, 2)
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.status = status
        if attributes:
            self.attributes.update(attributes)

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
            "attributes": self.attributes,
            "tool_calls": self.tool_calls,
            "events": self.events,
        }


class ObservabilityTracer:
    """
    End-to-End Multi-Agent Distributed Tracer.
    Captures spans, tool invocations, confidence drift across layers, and latency waterfalls.
    """

    def __init__(self, claim_id: Optional[str] = None):
        self.trace_id: str = f"trace_{uuid.uuid4().hex[:16]}"
        self.claim_id: str = claim_id or f"claim_{uuid.uuid4().hex[:8]}"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.start_Perf: float = time.perf_counter()
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
                    },
                },
            )

    def build_summary(self, decision_trace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_duration_ms = round((time.perf_counter() - self.start_Perf) * 1000.0, 2)
        tool_call_count = sum(len(s.tool_calls) for s in self.spans)

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
            "confidence_progression": confidence_progression,
            "outcome_progression": outcome_progression,
            "spans": [s.to_dict() for s in self.spans],
        }
