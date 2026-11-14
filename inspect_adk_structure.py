import google.adk
import google.adk.models
import inspect

print(f"ADK Location: {google.adk.__file__}")
print(f"ADK Version: {getattr(google.adk, '__version__', 'unknown')}")

print("\n--- google.adk Members ---")
print(dir(google.adk))

print("\n--- google.adk.models Members ---")
print(dir(google.adk.models))

# Check inside google.adk.types if it exists
try:
    import google.adk.types
    print("\n--- google.adk.types Members ---")
    print(dir(google.adk.types))
except ImportError:
    print("\n[!] google.adk.types does not exist")

# Check inside google.adk.events
try:
    import google.adk.events
    print("\n--- google.adk.events Members ---")
    print(dir(google.adk.events))
except ImportError:
    print("\n[!] google.adk.events does not exist")
