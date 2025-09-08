"""Main entry point for the Autonomous ML Agent."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import after path modification
from cli import cli  # noqa: E402

if __name__ == "__main__":
    cli()
