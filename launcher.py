"""
Desktop launcher for the Tamil Nadu Climate-Responsive Seed & Irrigation
Advisor (Streamlit app). This is the file PyInstaller packages into a .exe
— it starts the Streamlit server in-process and opens the default browser
automatically, so the end user just double-clicks and the app appears.

Folder layout expected next to this file (and next to the built .exe):
    streamlit_app.py
    outputs/catboost_crop_model.cbm
    nasa_power_tamilnadu/tamilnadu_climate_monthly_agg_2015_2024.csv
"""

import os
import sys
import threading
import webbrowser
import time

from streamlit.web import cli as stcli


def resource_path(relative_path):
    """Resolve a path that works both when run as a plain script and
    when bundled by PyInstaller (which unpacks to a temp _MEIPASS folder)."""
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def open_browser():
    time.sleep(2.5)  # give the Streamlit server a moment to start
    webbrowser.open("http://localhost:8501")


if __name__ == "__main__":
    app_path = resource_path("streamlit_app.py")

    threading.Thread(target=open_browser, daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())
