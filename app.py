import sys
import os
from streamlit.web import cli as stcli


def main():
    # Default to 127.0.0.1 for local/test security; allow override via SERVER_ADDRESS in Cloud Run container
    default_host = "127.0.0.1"
    host = os.getenv("SERVER_ADDRESS", default_host)
    port = os.getenv("PORT", "8080")
    sys.argv = [
        "streamlit",
        "run",
        "ui/dashboard.py",
        f"--server.port={port}",
        f"--server.address={host}",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

