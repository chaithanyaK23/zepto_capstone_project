"""Run the full Q2 analytics pipeline in the required order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent


def run_step(script_name: str) -> None:
    """Run one pipeline step with the same Python interpreter."""
    script_path = MODULE_DIR / script_name
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    """Run EDA first, then modeling, so Titanic is loaded only once."""
    run_step("01_eda.py")
    run_step("02_modeling.py")
    print("\nQ2 analytics pipeline completed successfully.")


if __name__ == "__main__":
    main()
