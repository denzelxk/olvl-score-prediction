"""
app/__init__.py
Ensures the project root (parent of this directory) is on sys.path so that
`from app.utils.xxx import ...` and `from src.xxx import ...` resolve correctly
regardless of the working directory when Streamlit is launched.
"""
import sys, os

_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)