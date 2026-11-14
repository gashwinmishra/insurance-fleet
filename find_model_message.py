import google.adk
import pkgutil
import importlib
import inspect

print("Searching for ModelMessage in google.adk...")

# Recursive search helper
def search_module(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception:
        return

    # Check members
    if hasattr(mod, "ModelMessage"):
        print(f"FOUND: {module_name}.ModelMessage")
    
    # Recurse
    if hasattr(mod, "__path__"):
        for _, sub_name, _ in pkgutil.iter_modules(mod.__path__):
            search_module(f"{module_name}.{sub_name}")

search_module("google.adk")
