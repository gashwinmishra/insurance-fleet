
import os
from google.adk.models import Gemini
import google.auth

# Auth is auto-handled by ADK usually, or needs explicit setup
project_id = os.getenv("PROJECT_ID")
location = "us-central1"

print(f"Testing Gemini ADK with Project: {project_id}")

try:
    model = Gemini(
        model_name="gemini-1.5-flash-001",
        project_id=project_id,
        location=location
    )
    
    print("Model initialized.")
    response = model.generate_content("Hello, can you hear me?")
    print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {e}")
