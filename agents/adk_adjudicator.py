import json
import os
import uuid
from types import SimpleNamespace
from typing import Any, Dict, Optional

import google.auth
from google.adk import Agent, Runner
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from vertexai.generative_models import Part
from google.genai import Client
from google.genai import types

from agents.adjudicator import AdjudicatorAgent
from agents.tools import (
    ADK_GENAI_TOOLS,
    AdjudicationTraceSchema,
    lookup_official_policy,
    retrieve_institutional_knowledge,
    retrieve_knowledge,
    query_override_telemetry,
    propose_policy_amendment,
    validate_adjudication_trace,
)


class ModelRouter:
    """
    Strategic Model Router (Pillar 3: Orchestration & Logic).
    Dynamically routes tasks between fast/cost-efficient models (`gemini-2.0-flash-001`)
    for perception and routine claims, and deep reasoning models (`gemini-1.5-pro-002`)
    for high-stakes safety hazards, policy-vs-RAG conflict resolution, and Layer 3
    Slow-Loop policy amendment synthesis.
    """

    FLASH_MODEL = "gemini-2.0-flash-001"
    PRO_MODEL = "gemini-1.5-pro-002"

    HIGH_STAKES_DAMAGE = {"puncture", "corrosion", "granule_loss", "missing_material", "chip"}
    HIGH_STAKES_LOCATIONS = {"battery_casing", "hv_cable_junction", "glass_center", "roof_surface"}

    @classmethod
    def route(
        cls,
        visual_findings: Dict[str, Any],
        claim_type: str = "auto",
        task_stage: str = "tiered_adjudication",
    ) -> Dict[str, str]:
        """
        Determines the optimal Gemini model tier based on task stage, severity,
        damage category, and potential policy override complexity.
        """
        if task_stage == "visual_perception":
            return {
                "selected_model": cls.FLASH_MODEL,
                "tier": "FAST_MULTIMODAL_PERCEPTION",
                "reason": "Low-latency multimodal feature extraction routed to Gemini Flash.",
            }

        severity = str(visual_findings.get("severity", "minor")).lower()
        damage_type = str(visual_findings.get("damage_type", "unknown")).lower()
        location = str(visual_findings.get("location", "")).lower()
        area_pct = int(visual_findings.get("affected_area_percentage", 0) or 0)

        is_high_severity = severity in {"critical", "major", "high", "total_loss"}
        is_high_stakes = (
            damage_type in cls.HIGH_STAKES_DAMAGE
            or any(loc in location for loc in cls.HIGH_STAKES_LOCATIONS)
            or "battery" in location
            or area_pct >= 20
        )

        if is_high_severity or is_high_stakes or task_stage == "slow_loop_policy_synthesis":
            return {
                "selected_model": cls.PRO_MODEL,
                "tier": "PRO_COMPLEX_PLANNING",
                "reason": (
                    f"Complex multi-layer conflict or high-stakes claim "
                    f"(damage={damage_type}, location={location}, severity={severity}) "
                    f"routed to {cls.PRO_MODEL} for deep reasoning and Slow-Loop policy synthesis."
                ),
            }

        return {
            "selected_model": cls.FLASH_MODEL,
            "tier": "FLASH_ROUTINE_ADJUDICATION",
            "reason": (
                f"Standard low-severity claim (damage={damage_type}, severity={severity}) "
                f"routed to {cls.FLASH_MODEL} for low-latency adjudication."
            ),
        }


class AdkAdjudicatorAgent:
    """
    Google ADK Multi-Layer Adjudicator Agent with Strategic Model Routing & Formal JSON Schemas.
    Maintains both a Fast Tier (`gemini-2.0-flash-001`) and Complex Reasoning Tier (`gemini-1.5-pro-002`)
    ADK Agent/Runner pair, enforcing `AdjudicationTraceSchema` and `ADK_GENAI_TOOLS` constraints.
    """

    SYSTEM_INSTRUCTION = """
    You are a Senior Insurance Adjudicator System.
    Your goal is to evaluate a claim based on Visual Findings across 3 cognitive layers.

    Use your tools systematically:
    1. Call `lookup_official_policy` to check the codified Base Policy terms (Layer 1).
    2. Call `retrieve_institutional_knowledge` to search internal safety alerts, memos, and vendor rules (Layer 2 Fast Loop).
    3. Call `query_override_telemetry` and `propose_policy_amendment` when a Fast Loop override occurs to draft a policy diff (Layer 3 Slow Loop).
    If any tool returns `"status": "ERROR"`, follow its `"recovery_guidance"` field immediately to self-correct your tool call.

    Return a strictly valid JSON object conforming to `AdjudicationTraceSchema`:
    {
        "layer_1_base": {"outcome": "...", "reason": "...", "source": "Official Policy", "confidence": float},
        "layer_2_inner": {"decision": {"outcome": "...", "reason": "...", "source": "...", "confidence": float}, "context_metadata": {"rag_override_reason": "...", "retrieved_rules": [{"author": "...", "source": "...", "content": "..."}]}},
        "layer_3_outer": {"decision": {"outcome": "...", "confidence": float, "source": "..."}, "signals": {"draft_policy_diff": "...", "frequency_stats": {"metric": "...", "value": "...", "trend": "..."}}}
    }

    IMPORTANT: The response MUST be strictly valid JSON. Escape any double quotes inside "draft_policy_diff".
    CRITICAL: ALWAYS populate "layer_3_outer" with a "draft_policy_diff" if you overrode the base decision.
    """

    def __init__(self):
        self.deterministic_engine = AdjudicatorAgent()
        self.router = ModelRouter()
        self._init_adk()

    def _init_adk(self):
        project_id = os.getenv("PROJECT_ID") or os.getenv("TC_PROJECT_ID")
        if not project_id:
            try:
                _, project_id = google.auth.default()
            except Exception:
                project_id = None

        if not project_id:
            project_id = "dummy-project"

        self.project_id = project_id

        # Tier 1: Fast Model Configuration (Gemini Flash)
        self.flash_model_config = Gemini(
            model=ModelRouter.FLASH_MODEL,
            project=project_id,
            location="us-central1",
        )
        # Tier 2: Complex Planning & Conflict Resolution Model Configuration (Gemini Pro)
        self.pro_model_config = Gemini(
            model=ModelRouter.PRO_MODEL,
            project=project_id,
            location="us-central1",
        )
        self.model_config = self.flash_model_config

        try:
            self.output_client = Client(vertexai=True, project=project_id, location="us-central1")
            self.flash_model_config.api_client = self.output_client
            self.pro_model_config.api_client = self.output_client
        except Exception:
            self.output_client = None

        tool_list = [
            lookup_official_policy,
            retrieve_institutional_knowledge,
            query_override_telemetry,
            propose_policy_amendment,
        ]

        self.flash_agent = Agent(
            name="adjudicator_flash_agent",
            model=self.flash_model_config,
            instruction=self.SYSTEM_INSTRUCTION,
            tools=tool_list,
        )
        self.pro_agent = Agent(
            name="adjudicator_pro_agent",
            model=self.pro_model_config,
            instruction=self.SYSTEM_INSTRUCTION,
            tools=tool_list,
        )
        self.agent = self.pro_agent

        self.session_service = InMemorySessionService()
        self.flash_runner = Runner(
            agent=self.flash_agent,
            session_service=self.session_service,
            app_name="insurance-fleet-adjudicator",
        )
        self.pro_runner = Runner(
            agent=self.pro_agent,
            session_service=self.session_service,
            app_name="insurance-fleet-adjudicator",
        )
        self.runner = self.pro_runner

    def adjudicate_tiered(
        self,
        visual_findings: Dict[str, Any],
        claim_type: str,
        session_id: Optional[str] = None,
        target_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Public API: Run Strategically Routed ADK 3-Layer Adjudicator with formal JSON Schema
        constraints (`AdjudicationTraceSchema`) and deterministic fallback.
        """
        routing_decision = self.router.route(visual_findings, claim_type=claim_type)
        selected_model = target_model or routing_decision["selected_model"]

        if self.project_id == "dummy-project" or os.getenv("ADK_OFFLINE_MODE") == "1":
            print(
                f"[AdkAdjudicator] Offline Mode active (Routed Model: {selected_model} "
                f"[{routing_decision['tier']}]). Using deterministic 3-Loop Engine."
            )
            trace = self.deterministic_engine.adjudicate_tiered(visual_findings, claim_type)
            validated = validate_adjudication_trace(trace)
            validated["model_routing"] = routing_decision
            return validated

        prompt = f"""
        Evaluate Claim:
        - Claim Type: {claim_type}
        - Object: {visual_findings.get('object_detected')}
        - Damage: {visual_findings.get('damage_type')}
        - Location: {visual_findings.get('location')}
        - Severity: {visual_findings.get('severity')}
        - Depth: {visual_findings.get('depth_mm')}
        - Affected Area: {visual_findings.get('affected_area_percentage')}%
        """

        active_runner = (
            self.pro_runner if selected_model == ModelRouter.PRO_MODEL else self.flash_runner
        )
        print(
            f"[AdkAdjudicator] Strategic Routing selected {selected_model} "
            f"({routing_decision['tier']}) for {visual_findings.get('damage_type')}..."
        )
        response_text = ""

        part = Part.from_text(prompt)
        message = SimpleNamespace(role="user", parts=[part])

        active_session_id = session_id or f"session_{uuid.uuid4()}"
        try:
            self.session_service.create_session_sync(
                user_id="user",
                session_id=active_session_id,
                app_name="insurance-fleet-adjudicator",
            )
            for event in active_runner.run(
                new_message=message, user_id="user", session_id=active_session_id
            ):
                try:
                    if getattr(event, "text", None):
                        response_text += event.text
                    elif hasattr(event, "candidates"):
                        cands = getattr(event, "candidates", [])
                        if cands:
                            cand_part = cands[0].content.parts[0]
                            if hasattr(cand_part, "text") and cand_part.text:
                                response_text += cand_part.text
                except Exception:
                    pass
        except Exception as e:
            print(f"[AdkAdjudicator] Runner Loop Error ({selected_model}): {e}")

        if (not response_text or not response_text.strip()) and self.output_client is not None:
            print(f"[AdkAdjudicator] Using Direct Client with formal JSON Schema on {selected_model}.")
            try:
                full_prompt = f"{self.SYSTEM_INSTRUCTION}\n\n{prompt}"
                direct_response = self.output_client.models.generate_content(
                    model=selected_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=AdjudicationTraceSchema,
                        tools=ADK_GENAI_TOOLS,
                        temperature=0.1,
                    ),
                )
                if direct_response.text:
                    response_text = direct_response.text
            except Exception as e:
                print(f"[AdkAdjudicator] Direct Client Fallback Failed: {e}")

        if response_text and response_text.strip():
            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_text)
                validated = validate_adjudication_trace(data)
                validated["model_routing"] = routing_decision
                return validated
            except Exception as e:
                print(f"[AdkAdjudicator] Error parsing LLM JSON ({e}). Delegating to deterministic 3-Loop Engine.")

        fallback_trace = self.deterministic_engine.adjudicate_tiered(visual_findings, claim_type)
        validated_fallback = validate_adjudication_trace(fallback_trace)
        validated_fallback["model_routing"] = routing_decision
        return validated_fallback
