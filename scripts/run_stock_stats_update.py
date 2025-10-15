#!/usr/bin/env python3
"""Standalone runner to execute `run_stock_stats_update` once."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aps import run_stock_stats_update, setup_logging  # noqa: E402
from utils.env import load_env_if_present  # noqa: E402


def main() -> None:
    setup_logging()
    load_env_if_present()
    run_stock_stats_update()


if __name__ == "__main__":
    main()
