"""Package-level alias for the ingest service (kept for backward compatibility)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingest import run_ingest, on_message, on_connect  # noqa: E402,F401

if __name__ == "__main__":
    run_ingest()
