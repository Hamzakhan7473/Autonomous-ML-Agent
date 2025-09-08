"""Main entry point for the Autonomous ML Agent."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from cli import cli

if __name__ == "__main__":
    cli()
