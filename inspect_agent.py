
from google.adk import Agent
import json

print("\n--- Agent Fields ---")
try:
    for name, field in Agent.model_fields.items():
        print(f"{name}: {field}")
except Exception as e:
    print(f"Not a Pydantic V2 model? {e}")
    try:
        # V1
        for name, field in Agent.__fields__.items():
            print(f"{name}: {field}")
    except:
        print("Not a Pydantic model")

print("\n--- Agent Help ---")
# help(Agent) # Output might be too long, let's just inspect init
print(Agent.__init__.__doc__)
