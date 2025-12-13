"""
Automation script to run the full pipeline for a specific account.
Order:
1. scripts/stock_meta_updater.py
2. scripts/update_price_cache.py
3. tune.py
4. backtest.py
5. recommend.py

Usage:
  python all.py <account_id>
"""

import subprocess
import sys
from pathlib import Path


def run_step(script_path: str, account_id: str):
    """Run a single script step with the given account ID."""
    # Ensure we use the same python interpreter
    cmd = [sys.executable, script_path, account_id]

    # Print clear separator and command info
    print(f"\n{'=' * 60}")
    print(f"[all.py] Running Step: {script_path} {account_id}")
    print(f"{'=' * 60}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[all.py] ❌ Error: Failed to execute {script_path}")
        print(f"[all.py] Exit Code: {e.returncode}")
        print("[all.py] Pipeline aborted.")
        sys.exit(e.returncode)


def main():
    if len(sys.argv) < 2:
        print("Usage: python all.py <account_id>")
        sys.exit(1)

    account_id = sys.argv[1]

    # List of scripts to run in order
    steps = [
        "scripts/stock_meta_updater.py",
        "scripts/update_price_cache.py",
        "tune.py",
        "backtest.py",
        "recommend.py",
    ]

    print(f"[all.py] Starting pipeline for account: {account_id}")
    print(f"[all.py] Steps: {', '.join(steps)}")

    for step in steps:
        # Check if file exists
        if not Path(step).exists():
            print(f"\n[all.py] ❌ Error: Script file not found: {step}")
            sys.exit(1)

        run_step(step, account_id)

    print(f"\n{'=' * 60}")
    print(f"[all.py] ✅ All steps completed successfully for {account_id}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
