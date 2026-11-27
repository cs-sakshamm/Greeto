"""
Shared pytest fixtures.
"""

from __future__ import annotations

import os
import pytest

# On Windows, onnxruntime / vision_service must be imported BEFORE PyQt5
# to ensure C++ / OpenMP DLL load order succeeds without access violations.
try:
    import app.vision.vision_service  # noqa: F401
except Exception:
    pass

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session")
def app():
    """Session-scoped QApplication instance for offscreen Qt tests."""
    qapp = QApplication.instance()
    if qapp is None:
        qapp = QApplication([])
    return qapp
