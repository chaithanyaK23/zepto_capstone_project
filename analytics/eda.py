"""Compatibility wrapper for the ordered Q2 EDA script.

Run `python run_all.py` from this folder for the full Q2 pipeline.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("01_eda.py")), run_name="__main__")
