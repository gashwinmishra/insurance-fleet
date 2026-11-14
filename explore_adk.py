
try:
    import google.adk
    print("Imported google.adk")
    print(dir(google.adk))
except ImportError:
    print("Failed to import google.adk")

try:
    import adk
    print("Imported adk")
    print(dir(adk))
except ImportError:
    print("Failed to import adk")
