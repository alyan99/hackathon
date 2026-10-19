"""
Hugging Face Space entry point — SaayaShift

This file is the standard HF Spaces entry point (app.py at repo root).
It simply delegates to src/app/app.py.
"""
import sys
from pathlib import Path

# Add project root to path so src.* imports resolve
sys.path.insert(0, str(Path(__file__).parent))

from src.app.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
