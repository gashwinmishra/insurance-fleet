from google.adk.models import LlmRequest
import inspect

print("--- LlmRequest Help ---")
print(help(LlmRequest))

print("\n--- LlmRequest Init Signature ---")
print(inspect.signature(LlmRequest.__init__))
