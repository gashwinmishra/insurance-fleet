import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Mock Env
os.environ["PROJECT_ID"] = "ashwin-genai-sandbox"
os.environ["TC_PROJECT_ID"] = "ashwin-genai-sandbox"

print("--- Importing AdkAdjudicatorAgent ---")
try:
    from agents.adk_adjudicator import AdkAdjudicatorAgent
except ImportError as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

print("--- Initializing Agent ---")
try:
    agent = AdkAdjudicatorAgent()
    print("Optimization: Agent Initialized (Using Internal Client Config)")
    print(f"Client VertexAI Status: {agent.model_config.api_client.vertexai}")
    
    print("\n--- TEST: Raw Client Generation ---")
    try:
        # Verify the client works directly
        test_prompt = "Hello, assume this is a test."
        print(f"Sending raw request to {agent.model_config.model}...")
        raw_response = agent.model_config.api_client.models.generate_content(
            model=agent.model_config.model,
            contents=test_prompt
        )
        print(f"Raw Response Text: {raw_response.text}")
    except Exception as e:
        print(f"Raw Client Failed: {e}")

except Exception as e:
    # Also need to update agent.agent.model? No, agent.agent.model IS agent.model_config usually.
    # Let's check init of AdkAdjudicatorAgent.
    # self.agent = Agent(..., model=self.model_config)
    # The Agent likely holds a reference to model_config.
    # So updating agent.model_config.api_client should work IF Agent uses it.
    
    print("Optimization: Agent Initialized & Client Injected")
    print(f"Client VertexAI Status: {agent.model_config.api_client.vertexai}")
except Exception as e:
    print(f"Init Failed: {e}")
    sys.exit(1)

print("--- Running Adjudication ---")
sample_findings = {
    "object_detected": "HV Connector",
    "damage_type": "corrosion",
    "location": "battery_system",
    "severity": "moderate",
    "depth_mm": 0.0,
    "affected_area_percentage": 10
}

try:
    result = agent.adjudicate_tiered(sample_findings, "auto")
    print("\n--- Result ---")
    print(result)
except Exception as e:
    print(f"\n[!] Execution Failed: {e}")
    import traceback
    traceback.print_exc()
