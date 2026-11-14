from google.adk.models import LlmRequest

try:
    req = LlmRequest(role="user", parts=["test"])
    print("Success: LlmRequest accepts role and parts")
    print(req)
except Exception as e:
    print(f"Failed: {e}")
    # Inspect fields
    print("LlmRequest fields:", LlmRequest.model_fields.keys())
