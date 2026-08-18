"""
Root entry point — used both for normal development runs and as the
target script for PyInstaller when building the Windows .exe.

Run with:  python main.py
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from gui.main_window import MainWindow


if __name__ == "__main__":
    # Fixes text/label clipping on Windows displays scaled above 100%.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    w = MainWindow()
    w.show()

    sys.exit(app.exec())
