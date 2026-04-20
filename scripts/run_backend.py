"""Run the backend server.

Usage:
    python -m scripts.run_backend [--reload]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from src.config.settings import get_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=args.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
