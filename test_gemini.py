
import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not set.")
    exit(1)

print(f"Using Key: {api_key[:5]}...{api_key[-5:]}")

try:
    genai.configure(api_key=api_key)
    print("Listing models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    print("\nTesting gemini-1.5-flash-001...")
    model = genai.GenerativeModel('gemini-1.5-flash-001')
    response = model.generate_content("Hello")
    print(f"Success! Response: {response.text}")

except Exception as e:
    print(f"\nFATAL ERROR: {e}")
