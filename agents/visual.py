import json
import os

class VisualAgent:
    def __init__(self):
        # Load scenarios map
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scenarios_path = os.path.join(base_path, 'data/scenarios.json')
        self.scenarios = self._load_scenarios()

    def _load_scenarios(self):
        try:
            with open(self.scenarios_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {"auto": [], "home": []}

    def analyze(self, image_input, context="auto"):
        """
        Simulates visual analysis of an image by looking up the scenario.
        image_input: expected to be the filename or path.
        """
        print(f"[VisualAgent] Scanning image for context: {context}...")
        
        # Extract filename from path if needed
        filename = os.path.basename(image_input)
        
        # Look for matching scenario
        dataset = self.scenarios.get(context, [])
        for case in dataset:
            if case['image_file'] == filename:
                print(f"[VisualAgent] Matched scenario: {case['title']}")
                return case['visual_findings']
        
        # Fallback default if no match found (for safety)
        # Fallback: Live Gemini Inference for User Uploads
        return self._analyze_with_gemini(image_input, context)

    def _analyze_with_gemini(self, image_path, context):
        """
        Calls Gemini Pro Vision via Vertex AI SDK (Enterprise).
        Uses ADC (Application Default Credentials).
        """
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, Part, SafetySetting
            import google.auth
            
            # Auto-detect project if not set
            project_id = os.getenv("TC_PROJECT_ID") or os.getenv("PROJECT_ID")
            location = os.getenv("TC_LOCATION", "us-central1")
            
            if not project_id:
                # Try to get default project from gcloud auth
                _, project_id = google.auth.default()
            
            if not project_id:
                return {
                    "object_detected": "Configuration Error",
                    "damage_type": "unknown",
                    "visual_evidence": "Could not determine Google Cloud Project ID. Set PROJECT_ID env var.",
                    "severity": "unknown"
                }

            vertexai.init(project=project_id, location=location)
            
            # Load Model (Gemini 1.5 Flash is GA on Vertex)
            model = GenerativeModel("gemini-1.5-flash-001")

            if not os.path.exists(image_path):
                 return {"visual_evidence": f"Error: File not found at {image_path}"}

            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Create Part from bytes
            image_part = Part.from_data(data=image_data, mime_type="image/jpeg")
            
            prompt = f"""
            You are an expert insurance adjuster AI. Analyze this image for a {context} insurance claim.
            Return ONLY a valid JSON object with the following fields:
            - object_detected: (str) Make/Model of car or type of home structure
            - damage_type: (str) e.g., scratch, dent, crack, shingle_damage
            - location: (str) e.g., front_bumper, roof_slope_north, battery_casing
            - severity: (str) low, medium, high, or total_loss
            - depth_mm: (int) estimated depth of damage in mm (use 0 if not applicable)
            - affected_area_percentage: (int) 0-100 (mainly for home/roof)
            - visual_evidence: (str) A brief professional summary of the visual findings.
            
            Do not include markdown backticks. Just the raw JSON string.
            """
            
            response = model.generate_content([prompt, image_part])
            
            # Clean response
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        except Exception as e:
            print(f"[VisualAgent] Vertex AI Inference Failed: {e}")
            return {
                "object_detected": "Vertex AI Error",
                "damage_type": "unknown",
                "visual_evidence": f"System Error: {str(e)}",
                "severity": "unknown"
            }
