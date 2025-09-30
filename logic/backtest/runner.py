"""Backtest runner extracted from test.py

Provides a callable `run(account: str, quiet: bool = False, prefetched_data=None, override_settings=None)`
that delegates to `test.main` for now. This keeps CLI decoupled from module path.
"""
from __future__ import annotations

import os
import sys

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Dict

import pandas as pd  # noqa: F401  # kept for potential future extensions

from test import main as _run_backtest_main


def run(
    account: str,
    quiet: bool = False,
    prefetched_data: Optional[Dict] = None,
    override_settings: Optional[Dict] = None,
):
    """Run backtest by delegating to original test.main."""
    return _run_backtest_main(
        account=account,
        quiet=quiet,
        prefetched_data=prefetched_data,
        override_settings=override_settings,
    )


__all__ = ["run"]
