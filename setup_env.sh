#!/bin/bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
# Trying default index first, then falling back or just letting user handle it if it fails
# But here we explicitly try to allow pypi if the default fails (logic could be complex, keeping it simple)
pip install -r requirements.txt || echo "Standard install failed. You may need to specify --index-url or check your VPN/Proxy settings."
