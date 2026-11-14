
import os
import asyncio
import google.auth
from google.adk import Agent
from google.adk.models import Gemini

async def main():
    creds, project_id = google.auth.default()
    print(f"Project: {project_id}")

    # Configure Model
    model_config = Gemini(
        model="gemini-1.5-flash-001",
        project_id=project_id,
        location="us-central1"
    )

    # Initialize Agent
    agent = Agent(
        name="test_agent",
        model=model_config,
        instruction="You are a helpful assistant."
    )

    print("Agent initialized. Running...")
    
    # Run Agent
    # Agent might use 'run', 'prompt', 'ask'? 
    # Let's inspect it first
    print("\n[Dir Agent]:", dir(agent))
    
    # Try run_async
    try:
        response_text = ""
        async for chunk in agent.run_async("Hello!"):
             # Chunk might be Content or str
             print(f"Chunk Type: {type(chunk)}")
             # Assuming chunk has text or is text
             if hasattr(chunk, 'text'):
                 response_text += chunk.text
             else:
                 response_text += str(chunk)
                 
        print(f"Full Response: {response_text}")
    except Exception as e:
        print(f"Run Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
