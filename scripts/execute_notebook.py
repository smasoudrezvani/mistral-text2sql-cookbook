"""Execute the cookbook notebook end to end and save it with its outputs.

Runs in a temporary folder (the notebook downloads ~19 MB of databases there), with
MISTRAL_API_KEY taken from .env. Costs a few cents (30 questions x 3 approaches, Small 4).

Usage:
    uv run python scripts/execute_notebook.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nbformat
from dotenv import load_dotenv
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks/text2sql_mistral_bird.ipynb"


def main() -> None:
    """Execute the notebook and overwrite it with the executed version."""
    load_dotenv(ROOT / ".env")
    if not os.environ.get("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY missing in .env")
    nb = nbformat.read(NB, as_version=4)
    with tempfile.TemporaryDirectory() as tmp:
        NotebookClient(
            nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": tmp}}
        ).execute()
    nbformat.write(nb, NB)
    print("executed and saved", NB)


if __name__ == "__main__":
    main()
