import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional

from agents.adk_visual import AdkVisualAgent
from agents.adk_adjudicator import AdkAdjudicatorAgent, ModelRouter
from agents.tools import (
    lookup_official_policy,
    retrieve_institutional_knowledge,
    query_override_telemetry,
    propose_policy_amendment,
    validate_visual_findings,
    validate_adjudication_trace,
)
from infrastructure.bq_logger import BQLogger
from infrastructure.memory_store import MemoryStore
from infrastructure.observability import ObservabilityTracer
from infrastructure.rag_retriever import RAGRetriever


class ClaimOrchestrator:
    """
    Multi-Agent Workflow Orchestrator with Strategic Model Routing & Async Support (Pillar 3).
    Coordinates:
      1. Intake & Triage Guardrail
      2. Multimodal Perception (`AdkVisualAgent` on `gemini-2.0-flash-001`)
      3. Strategic Model Routing (`ModelRouter` selecting `gemini-1.5-pro-002` for complex/high-stakes
         reasoning & Slow-Loop policy synthesis vs `gemini-2.0-flash-001` for routine claims)
      4. 3-Loop Cognitive Adjudication (`AdkAdjudicatorAgent`)
      5. Safety & Confidence Escalation Gate
      6. Persistent SQLite/JSON Memory & PII-Scrubbed BigQuery Telemetry (`arun` & `run`)
    """

    CONFIDENCE_THRESHOLD = 0.75

    def __init__(self, default_claim_type: str = "auto"):
        self.default_claim_type = default_claim_type
        self.visual_agent = AdkVisualAgent()
        self.adjudicator = AdkAdjudicatorAgent()
        self.router = ModelRouter()
        self.logger = BQLogger()
        self.memory = MemoryStore()
        self.rag = RAGRetriever()

    def run(self, image_input: str, claim_type: Optional[str] = None) -> Dict[str, Any]:
        domain = (claim_type or self.default_claim_type).lower().strip()
        if domain not in ("auto", "home"):
            domain = "auto"

        claim_id = str(uuid.uuid4())
        session_id = f"session_{claim_id[:8]}"
        tracer = ObservabilityTracer(claim_id=claim_id)

        # 1. Intake & Triage Guardrail
        with tracer.start_span(
            name="intake_and_triage",
            agent="ClaimOrchestrator",
            attributes={"claim_type": domain, "image_file": os.path.basename(str(image_input))},
        ) as span:
            self.memory.append_session_event(
                session_id=session_id,
                role="system",
                content={"stage": "intake", "claim_id": claim_id, "claim_type": domain},
            )
            span.add_event("session_initialized", {"session_id": session_id})

        # 2. Multi-Modal Visual Perception (Fast Tier: gemini-2.0-flash-001)
        perception_route = self.router.route({}, claim_type=domain, task_stage="visual_perception")
        with tracer.start_span(
            name="visual_perception",
            agent="AdkVisualAgent",
            attributes={"context": domain, "routed_model": perception_route["selected_model"]},
        ) as span:
            raw_findings = self.visual_agent.analyze(image_input, context=domain)
            findings = validate_visual_findings(raw_findings, claim_type=domain)
            span.finish(
                status="OK",
                attributes={
                    "object_detected": findings.get("object_detected"),
                    "damage_type": findings.get("damage_type"),
                    "severity": findings.get("severity"),
                    "routed_model": perception_route["selected_model"],
                },
            )
            self.memory.append_session_event(
                session_id=session_id,
                role="visual_agent",
                content=findings,
            )

        # 3. Strategic Model Routing (Flash vs Pro based on complexity/risk)
        adjudication_route = self.router.route(
            findings, claim_type=domain, task_stage="tiered_adjudication"
        )
        with tracer.start_span(
            name="strategic_model_routing",
            agent="ModelRouter",
            attributes=adjudication_route,
        ) as span:
            span.add_event("model_selected", adjudication_route)

        # 4. 3-Loop Cognitive Adjudication & Tool Instrumentation
        with tracer.start_span(
            name="tiered_adjudication",
            agent="AdkAdjudicatorAgent",
            attributes={
                "claim_type": domain,
                "damage_type": findings.get("damage_type"),
                "selected_model": adjudication_route["selected_model"],
                "routing_tier": adjudication_route["tier"],
            },
        ) as span:
            dmg = findings.get("damage_type", "unknown")
            loc = findings.get("location", "")
            query_str = f"{findings.get('object_detected', '')} {dmg} {loc}".strip()

            t0 = time.perf_counter()
            policy_out = lookup_official_policy(domain, dmg)
            span.record_tool_call(
                "lookup_official_policy",
                {"claim_type": domain, "damage_type": dmg},
                policy_out,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

            t1 = time.perf_counter()
            rag_out = retrieve_institutional_knowledge(query_str, claim_type=domain, top_k=3)
            span.record_tool_call(
                "retrieve_institutional_knowledge",
                {"query": query_str, "claim_type": domain, "top_k": 3},
                rag_out,
                duration_ms=(time.perf_counter() - t1) * 1000.0,
            )

            t2 = time.perf_counter()
            telemetry_out = query_override_telemetry(domain, dmg, loc)
            span.record_tool_call(
                "query_override_telemetry",
                {"claim_type": domain, "damage_type": dmg, "location": loc},
                telemetry_out,
                duration_ms=(time.perf_counter() - t2) * 1000.0,
            )

            raw_trace = self.adjudicator.adjudicate_tiered(
                findings,
                claim_type=domain,
                session_id=session_id,
                target_model=adjudication_route["selected_model"],
            )
            decision_trace = validate_adjudication_trace(raw_trace)

            diff_text = decision_trace.get("layer_3_outer", {}).get("signals", {}).get("draft_policy_diff")
            if diff_text:
                span.record_tool_call(
                    "propose_policy_amendment",
                    {"claim_type": domain, "damage_type": dmg},
                    diff_text,
                    duration_ms=0.5,
                )

            span.finish(
                status="OK",
                attributes={
                    "selected_model": adjudication_route["selected_model"],
                    "layer_1_outcome": decision_trace["layer_1_base"]["outcome"],
                    "layer_2_outcome": decision_trace["layer_2_inner"]["decision"]["outcome"],
                    "layer_3_outcome": decision_trace["layer_3_outer"]["decision"]["outcome"],
                },
            )
            self.memory.append_session_event(
                session_id=session_id,
                role="adjudicator_agent",
                content=decision_trace["layer_2_inner"]["decision"],
            )

        # 5. Safety & Confidence Escalation Gate
        with tracer.start_span(
            name="safety_and_confidence_gate",
            agent="EscalationGuardrail",
        ) as span:
            escalation = self._evaluate_escalation_gate(findings, decision_trace)
            span.finish(status="OK", attributes=escalation)

        # 6. Memory Persistence & BigQuery Audit Logging
        with tracer.start_span(
            name="telemetry_and_memory_commit",
            agent="BQLogger+MemoryStore",
        ) as span:
            self.memory.record_claim(
                claim_id=claim_id,
                claim_type=domain,
                findings=findings,
                decision_trace=decision_trace,
                trace_id=tracer.trace_id,
            )
            observability_summary = tracer.build_summary(decision_trace)
            self.logger.log_feedback(
                claim_id=claim_id,
                agent_decision=decision_trace["layer_2_inner"]["decision"],
                trace_id=tracer.trace_id,
                latency_ms=observability_summary["total_duration_ms"],
            )
            span.finish(status="OK", attributes={"trace_id": tracer.trace_id})

        final_trace_summary = tracer.build_summary(decision_trace)

        return {
            "claim_id": claim_id,
            "session_id": session_id,
            "claim_type": domain,
            "findings": findings,
            "decision_trace": decision_trace,
            "model_routing": adjudication_route,
            "escalation": escalation,
            "observability": final_trace_summary,
        }

    async def arun(self, image_input: str, claim_type: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronous non-blocking orchestration flow preventing UI event-loop blocking."""
        return await asyncio.to_thread(self.run, image_input, claim_type)

    def _evaluate_escalation_gate(
        self,
        findings: Dict[str, Any],
        decision_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluates safety hazards and confidence gates for human adjuster escalation."""
        l2_decision = decision_trace.get("layer_2_inner", {}).get("decision", {})
        outcome = str(l2_decision.get("outcome", "")).upper()
        confidence = float(l2_decision.get("confidence", 0.9))
        damage_type = str(findings.get("damage_type", "")).lower()
        location = str(findings.get("location", "")).lower()

        if "HAZMAT" in outcome or damage_type == "puncture" or ("battery" in location and "TOTAL_LOSS" in outcome):
            return {
                "requires_human_signoff": True,
                "status": "CRITICAL_SAFETY_ESCALATION",
                "reason": "High-voltage EV battery or Hazmat containment protocol triggered.",
                "confidence": confidence,
            }

        if confidence < self.CONFIDENCE_THRESHOLD or "MANUAL_REVIEW" in outcome or "REVIEW_REQUIRED" in outcome:
            return {
                "requires_human_signoff": True,
                "status": "LOW_CONFIDENCE_REVIEW",
                "reason": f"Confidence ({confidence:.2f}) below autonomous threshold or manual review required.",
                "confidence": confidence,
            }

        return {
            "requires_human_signoff": False,
            "status": "AUTONOMOUS_APPROVED",
            "reason": f"Confidence ({confidence:.2f}) meets autonomous execution threshold.",
            "confidence": confidence,
        }

    def submit_human_feedback(
        self,
        claim_id: str,
        claim_type: str,
        damage_type: str,
        original_outcome: str,
        corrected_outcome: str,
        adjuster_notes: str,
        adjuster_name: str = "Senior Adjuster",
        promote_to_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Records a Human-in-the-Loop (HITL) override across BigQuery, Episodic Memory,
        and optionally promotes the feedback into Institutional Knowledge (`RAGRetriever`).
        """
        override_record = self.memory.record_human_override(
            claim_id=claim_id,
            claim_type=claim_type,
            damage_type=damage_type,
            original_outcome=original_outcome,
            corrected_outcome=corrected_outcome,
            adjuster_notes=adjuster_notes,
            adjuster_name=adjuster_name,
        )

        self.logger.log_feedback(
            claim_id=claim_id,
            agent_decision={"outcome": original_outcome},
            human_correction={
                "corrected_outcome": corrected_outcome,
                "notes": adjuster_notes,
                "adjuster": adjuster_name,
            },
        )

        new_rule = None
        if promote_to_rag and adjuster_notes.strip():
            new_rule = self.rag.add_knowledge_rule(
                content=f"{damage_type.upper()} OVERRIDE ({corrected_outcome}): {adjuster_notes.strip()}",
                author=adjuster_name,
                source=f"HITL Override ({claim_type.upper()} Claims)",
                context=f"Claim #{claim_id[:8]} Correction",
                weight=0.92,
                persist=False,
            )

        return {
            "override_record": override_record,
            "promoted_rag_rule": new_rule,
        }
