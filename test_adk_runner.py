
import os
import google.auth
from google.adk import Agent
from google.adk.runners import Runner # Guessing import path or just google.adk.Runner
from google.adk.models import Gemini

def main():
    creds, project_id = google.auth.default()
    
    model = Gemini(
        model="gemini-1.5-flash-001",
        project_id=project_id,
        location="us-central1"
    )
    
    agent = Agent(
        name="test_agent",
        model=model,
        instruction="You are a helpful assistant."
    )
    
    # Try importing Runner from top level if possible
    # In explore_adk.py, 'Runner' was in dir(google.adk)
    try:
        from google.adk import Runner
        print("Imported Runner from google.adk")
    except ImportError:
        print("Imported Runner check failed")
        return

    try:
        from google.adk.sessions import InMemorySessionService
    except ImportError:
        print("Imported Session check failed")
        return

    runner = Runner(
        agent=agent, 
        session_service=InMemorySessionService(),
        app_name="visual-agent-app"
    )
    print("Runner initialized")
    
    # Try run methods
    # maybe runner.run("Hello")?
    print("Runner dir:", dir(runner))
    
    # Runner.run might be async? Or sync?
    # Let's check help or just try calling it synchronously first, then async if needed.
    # If Runner has run_async, we use that.
    if hasattr(runner, 'run'):
        result = runner.run(user_input="Hello!") # Guessing arg name 'user_input' or Positional
        print("Result:", result)
    elif hasattr(runner, 'run_async'):
        import asyncio
        result = asyncio.run(runner.run_async("Hello!"))
        print("Result:", result)

if __name__ == "__main__":
    main()
