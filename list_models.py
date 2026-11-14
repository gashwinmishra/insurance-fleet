from google.genai import Client
import os

project_id = "ashwin-genai-sandbox"
location = "us-central1"

print(f"Listing models for {project_id} in {location}...")
try:
    client = Client(vertexai=True, project=project_id, location=location)
    # Using list_models from keys if available, or just iterating defaults
    # The client might have a models.list method
    # Checking dir(client.models) would be good if I can't guess
    # standard genai is client.models.list() usually
    for m in client.models.list():
        print(f"Model: {m.name}")
        # print specific fields if interesting
except Exception as e:
    print(f"Failed to list models: {e}")
    # try verifying client structure
    import dir
    # print(dir(client)) 
