"""Compatibility entrypoint for the Carbon-24 Dashboard v2.

The full dashboard implementation is in ``carbon24_dashboard-v2.py``. This
wrapper keeps the underscore filename usable with Streamlit:

    streamlit run carbon24_dashboard_v2.py
"""

from pathlib import Path
import runpy


DASHBOARD_PATH = Path(__file__).with_name("carbon24_dashboard-v2.py")

if not DASHBOARD_PATH.exists():
    raise FileNotFoundError(f"Dashboard file not found: {DASHBOARD_PATH}")

runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")
