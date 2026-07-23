"""Ensure tests exercise the vendored RWARE source in this unified repository."""

import sys
from pathlib import Path


RWARE_SOURCE = Path(__file__).resolve().parent / "robotic-warehouse"
if str(RWARE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RWARE_SOURCE))
