import inspect
from google.adk.models import Gemini

print("--- Gemini Constructor Signature ---")
try:
    sig = inspect.signature(Gemini.__init__)
    print(sig)
except Exception as e:
    print(f"Could not get signature: {e}")

print("\n--- Gemini Docstring ---")
import google.adk.models
print(f"Module Source: {google.adk.models.__file__}")

print("\n--- Testing google.genai.Client ---")
import os
try:
    from google.genai import Client
    print("Init Client with vertexai=True...")
    client = Client(vertexai=True, project="ashwin-genai-sandbox", location="us-central1")
    print("Client Init Success")
except Exception as e:
    print(f"Client Init Failed: {e}")
