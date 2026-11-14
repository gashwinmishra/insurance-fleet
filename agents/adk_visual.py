import os
import json
import uuid
from types import SimpleNamespace
import google.auth
from google.adk import Agent, Runner
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from vertexai.generative_models import Part
from google.genai import Client
from agents.tools import validate_visual_findings


class AdkVisualAgent:
    def __init__(self):
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scenarios_path = os.path.join(base_path, "data/scenarios.json")
        self.scenarios = self._load_scenarios()
        self._init_adk()

    def _load_scenarios(self):
        try:
            with open(self.scenarios_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"auto": [], "home": []}

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
            name="visual_agent",
            model=self.model_config,
            instruction="You are an expert insurance adjuster AI. Return ONLY JSON.",
        )

        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name="insurance-fleet-visual",
        )

    def analyze(self, image_input, context="auto"):
        """
        Public API: Analyze image (Scenario Lookup or Live ADK Vision Inference).
        Guarantees a normalized VisualFindings schema contract.
        """
        print(f"[AdkVisualAgent] Scanning image for context: {context}...")

        filename = os.path.basename(str(image_input))
        dataset = self.scenarios.get(context, [])
        for case in dataset:
            if case["image_file"] == filename:
                print(f"[AdkVisualAgent] Matched scenario: {case['title']}")
                return validate_visual_findings(case["visual_findings"], claim_type=context)

        raw_result = self._analyze_with_adk(image_input, context)
        return validate_visual_findings(raw_result, claim_type=context)

    def _analyze_with_adk(self, image_path, context):
        try:
            if not os.path.exists(image_path):
                return {"visual_evidence": f"Error: File not found at {image_path}"}

            with open(image_path, "rb") as f:
                image_data = f.read()

            image_part = Part.from_data(data=image_data, mime_type="image/jpeg")

            prompt = f"""
            Analyze this image for a {context} insurance claim.
            Return ONLY a valid JSON object with:
            - object_detected, damage_type, location, severity, depth_mm, affected_area_percentage, visual_evidence.
            Do not include markdown backticks.
            """

            text_part = Part.from_text(prompt)
            message_content = SimpleNamespace(role="user", parts=[text_part, image_part])

            print("[AdkVisualAgent] Running ADK Inference...")
            response_text = ""

            session_id = f"session_{uuid.uuid4()}"
            self.session_service.create_session_sync(
                user_id="user",
                session_id=session_id,
                app_name="insurance-fleet-visual",
            )

            try:
                for event in self.runner.run(new_message=message_content, user_id="user", session_id=session_id):
                    try:
                        text = getattr(event, "text", None)
                        if text:
                            response_text += text
                        elif hasattr(event, "candidates") and event.candidates:
                            part = event.candidates[0].content.parts[0]
                            if hasattr(part, "text") and part.text:
                                response_text += part.text
                    except Exception:
                        pass
            except Exception as e:
                print(f"[AdkVisualAgent] Runner Loop Error: {e}")

            if (not response_text or not response_text.strip()) and self.output_client is not None:
                print("[AdkVisualAgent] Runner yielded no text. Using Direct Client Fallback.")
                try:
                    direct_response = self.output_client.models.generate_content(
                        model="gemini-2.0-flash-001",
                        contents=[prompt, image_part],
                    )
                    if direct_response.text:
                        response_text = direct_response.text
                except Exception as e:
                    print(f"[AdkVisualAgent] Direct Client Fallback Failed: {e}")

            if not response_text:
                return {"visual_evidence": "ADK returned no text."}

            text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except Exception as e:
            print(f"[AdkVisualAgent] ADK Failed: {e}")
            return {
                "object_detected": "ADK Error",
                "damage_type": "unknown",
                "visual_evidence": f"System Error: {str(e)}",
                "severity": "unknown",
            }

