import os
import sys
import json
import asyncio
from agents.adk_visual import AdkVisualAgent
from agents.adk_adjudicator import AdkAdjudicatorAgent

# Mock Env
os.environ["PROJECT_ID"] = "ashwin-genai-sandbox"
os.environ["TC_PROJECT_ID"] = "ashwin-genai-sandbox"

def verify_system():
    print("--- [1] Initializing Agents ---")
    try:
        visual_agent = AdkVisualAgent()
        adjudicator_agent = AdkAdjudicatorAgent()
        print("✅ Agents Initialized")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    print("\n--- [2] Testing Visual Agent (Simulation) ---")
    # We use a simulated image path that triggers the mock or ADK if file exists
    # Let's use a real file if possible, or mockup. 
    # The Visual Agent has simulation logic if filename matches scenarios.
    # Let's try to trigger the ADK path if possible, OR the simulation path if that's safer for "not broken" baseline.
    # Actually, the user complaint is "move to ADK broken everything". 
    # So we should test the ADK path if possible, or at least the logic flow.
    # But we don't have a real image handy maybe? 
    # Let's use a mock return for Visual to isolate Adjudicator, THEN test Visual if we have an image.
    
    # Actually, let's test Adjudicator first as that was the main error source ("No patterns detected").
    
    print("\n--- [3] Testing Adjudicator Agent (Golden Path) ---")
    visual_findings = {
        "object_detected": "Tesla Model Y Battery",
        "damage_type": "scratch",
        "location": "battery_casing",
        "severity": "low",
        "depth_mm": 1.5,
        "affected_area_percentage": 5
    }
    
    print(f"Input Findings: {visual_findings}")
    result = adjudicator_agent.adjudicate_tiered(visual_findings, "auto")
    
    print("\n--- [4] Adjudicator Result Analysis ---")
    print(json.dumps(result, indent=2))
    
    layer_3 = result.get("layer_3_outer", {})
    diff = layer_3.get("signals", {}).get("draft_policy_diff")
    
    if diff:
        print(f"\n✅ SUCCESS: Policy Diff Generated:\n{diff}")
    else:
        print("\n❌ FAILURE: No Policy Diff Generated.")
        # This confirms if the prompt update fixed it.

if __name__ == "__main__":
    verify_system()
