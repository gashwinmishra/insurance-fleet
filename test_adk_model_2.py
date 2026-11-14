
import os
import google.auth
from google.adk.models import Gemini

print("--- Testing Gemini ADK ---")

# Auth
try:
    creds, project_id = google.auth.default()
    print(f"Auth Success. Project: {project_id}")
except Exception as e:
    print(f"Auth Failed: {e}")
    project_id = "test-project"

try:
    model = Gemini(
        model_name="gemini-1.5-flash-001",
        project_id=project_id,
        location="us-central1"
    )
    
    print("\n[Dir Model]:", dir(model))
    
    # Try likely methods
    if hasattr(model, 'generate'):
        print("Has generate()")
    if hasattr(model, 'query'):
        print("Has query()")
        
except Exception as e:
    print(f"Error: {e}")
