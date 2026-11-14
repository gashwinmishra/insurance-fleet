
import sys
import os
import json
from agents.adk_adjudicator import AdkAdjudicatorAgent

# Mock findings similar to a real case
visual_findings = {
    "object_detected": "Tesla Model 3",
    "damage_type": "scratch",
    "location": "battery_casing",
    "severity": "medium",
    "depth_mm": 1.5,
    "affected_area_percentage": 5
}
claim_type = "auto"

print("Initializing Adjudicator Agent...")
try:
    agent = AdkAdjudicatorAgent()
    print("Running adjudication...")
    result = agent.adjudicate_tiered(visual_findings, claim_type)
    
    print("\n--- RESULT JSON ---")
    print(json.dumps(result, indent=2))
    
    print("\n--- CHECKING CONFIDENCE ---")
    layer_1 = result.get("layer_1_base", {})
    layer_3 = result.get("layer_3_outer", {}).get("decision", {})
    
    print(f"Layer 1 Confidence: {layer_1.get('confidence')}")
    print(f"Layer 3 Confidence: {layer_3.get('confidence')}")

except Exception as e:
    print(f"Execution failed: {e}")
