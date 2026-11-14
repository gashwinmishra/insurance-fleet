import google.adk
import inspect
import pkgutil

print("--- google.adk content ---")
print(dir(google.adk))

print("\n--- Submodules ---")
for loader, module_name, is_pkg in pkgutil.walk_packages(google.adk.__path__, google.adk.__name__ + "."):
    print(module_name)

print("\n--- Searching for ModelMessage ---")
def find_class(module, class_name):
    for name, obj in inspect.getmembers(module):
        if name == class_name:
            return f"Found {class_name} in {module.__name__}"
        if inspect.ismodule(obj) and obj.__name__.startswith("google.adk"):
            # This recursion might be infinite if not careful, but for this depth it's fine or just use the flat list above
            pass

# Try standard locations
try:
    from google.adk import ModelMessage
    print("Found in google.adk")
except ImportError:
    pass

try:
    from google.adk.model import ModelMessage
    print("Found in google.adk.model")
except ImportError:
    pass

# search in all submodules
import importlib
for loader, module_name, is_pkg in pkgutil.walk_packages(google.adk.__path__, google.adk.__name__ + "."):
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "ModelMessage"):
            print(f"Found ModelMessage in {module_name}")
    except Exception as e:
        pass
