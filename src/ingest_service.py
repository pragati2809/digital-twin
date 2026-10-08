"""
Modular Ingest Service in src package.
"""

from ingest import run_ingest, on_message, on_connect

if __name__ == "__main__":
    run_ingest()
