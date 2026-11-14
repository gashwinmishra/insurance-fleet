import os
import json
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
    lookup_official_policy,
    retrieve_institutional_knowledge,
    retrieve_knowledge,
    query_override_telemetry,
    propose_policy_amendment,
    validate_adjudication_trace,
)


class AdkAdjudicatorAgent:
    """
    Google ADK Multi-Layer Adjudicator Agent.
    Equipped with 4 domain tools across the 3 cognitive loops:
      - Layer 1 (Base Policy): `lookup_official_policy`
      - Layer 2 (Fast Loop / RAG): `retrieve_institutional_knowledge`
      - Layer 3 (Slow Loop / Evolution): `query_override_telemetry`, `propose_policy_amendment`
    Includes a deterministic 3-loop fallback (`AdjudicatorAgent`) when executed in offline/CI environments.
    """

    def __init__(self):
        self.deterministic_engine = AdjudicatorAgent()
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

        self.model_config = Gemini(
            model="gemini-2.0-flash-001",
            project=project_id,
            location="us-central1",
        )

        try:
            self.output_client = Client(vertexai=True, project=project_id, location="us-central1")
            self.model_config.api_client = self.output_client
        except Exception:
            self.output_client = None

        self.agent = Agent(
            name="adjudicator_agent",
            model=self.model_config,
            instruction="""
            You are a Senior Insurance Adjudicator System.
            Your goal is to evaluate a claim based on Visual Findings across 3 cognitive layers.

            Use your tools systematically:
            1. Call `lookup_official_policy` to check the codified Base Policy terms (Layer 1).
            2. Call `retrieve_institutional_knowledge` to search internal safety alerts, memos, and vendor rules (Layer 2 Fast Loop).
            3. Call `query_override_telemetry` and `propose_policy_amendment` when a Fast Loop override occurs to draft a policy diff (Layer 3 Slow Loop).

            Return a strictly valid JSON object with this EXACT structure:
            {
                "layer_1_base": {"outcome": "...", "reason": "...", "source": "Official Policy", "confidence": float},
                "layer_2_inner": {"decision": {"outcome": "...", "reason": "...", "source": "...", "confidence": float}, "context_metadata": {"rag_override_reason": "...", "retrieved_rules": [{"author": "...", "source": "...", "content": "..."}]}},
                "layer_3_outer": {"decision": {"outcome": "...", "confidence": float, "source": "..."}, "signals": {"draft_policy_diff": "...", "frequency_stats": {"metric": "...", "value": "...", "trend": "..."}}}
            }

            IMPORTANT: The response MUST be strictly valid JSON. Escape any double quotes inside "draft_policy_diff".
            CRITICAL: ALWAYS populate "layer_3_outer" with a "draft_policy_diff" if you overrode the base decision.
            """,
            tools=[
                lookup_official_policy,
                retrieve_institutional_knowledge,
                query_override_telemetry,
                propose_policy_amendment,
            ],
        )

        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="insurance-fleet-adjudicator",
        )

    def adjudicate_tiered(
        self,
        visual_findings: Dict[str, Any],
        claim_type: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Public API: Run ADK 3-Layer Adjudicator with schema validation and deterministic fallback.
        """
        if self.project_id == "dummy-project" or os.getenv("ADK_OFFLINE_MODE") == "1":
            print("[AdkAdjudicator] Offline/Simulation Mode active. Using deterministic 3-Loop Engine.")
            trace = self.deterministic_engine.adjudicate_tiered(visual_findings, claim_type)
            return validate_adjudication_trace(trace)

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

        print(f"[AdkAdjudicator] Running logic for {visual_findings.get('damage_type')}...")
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
            for event in self.runner.run(new_message=message, user_id="user", session_id=active_session_id):
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
            print(f"[AdkAdjudicator] Runner Loop Error: {e}")

        if (not response_text or not response_text.strip()) and self.output_client is not None:
            print("[AdkAdjudicator] Runner yielded no text. Using Direct Client Fallback.")
            try:
                full_prompt = f"{self.agent.instruction}\n\n{prompt}"
                direct_response = self.output_client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                if direct_response.text:
                    response_text = direct_response.text
            except Exception as e:
                print(f"[AdkAdjudicator] Direct Client Fallback Failed: {e}")

        if response_text and response_text.strip():
            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_text)
                return validate_adjudication_trace(data)
            except Exception as e:
                print(f"[AdkAdjudicator] Error parsing LLM JSON ({e}). Delegating to deterministic 3-Loop Engine.")

        # Resilient fallback to deterministic 3-Loop Adjudicator
        fallback_trace = self.deterministic_engine.adjudicate_tiered(visual_findings, claim_type)
        return validate_adjudication_trace(fallback_trace)

