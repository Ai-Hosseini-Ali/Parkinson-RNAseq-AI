"""
Shared light clinical theme (blue / white / light-blue / navy) used across
main_window.py, explainability_panel.py, biomarker_panel.py, etc.

Keeping the palette and GlassPanel here means every screen re-themes
automatically when this file changes.
"""

from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt

# ---------------------------------------------------------------------------
# Palette — light clinical (blue / white / light-blue / navy)
# ---------------------------------------------------------------------------
NAVY_950 = "#EAF3FB"    # main app background (very light blue)
NAVY_900 = "#FFFFFF"    # pure white, spare
NAVY_800 = "#0B2545"    # deep navy — headers, primary buttons
NAVY_700 = "#134074"    # navy — gradients, secondary dark accents

BLUE_500 = "#2F6FED"    # primary blue accent
BLUE_400 = "#5B90F5"    # lighter blue accent

CYAN_400 = "#3FA9E0"    # sky-blue accent
CYAN_300 = "#8ED0F0"    # light-blue accent (badges, highlights)

ICE_100 = "#0B2545"     # primary text - deep navy (reads on white/light bg)
SLATE_300 = "#4A6079"   # secondary text
SLATE_400 = "#7C93AC"   # muted text / labels

GREEN_400 = "#1E8A5F"   # protective / healthy / success
AMBER_400 = "#B7791F"   # moderate / warning
RED_400 = "#C0392B"     # risk-increasing / danger

# Card surfaces
GLASS_FILL = "#F2F7FC"          # subtle hover fill
GLASS_FILL_STRONG = "#FFFFFF"   # solid white card background
GLASS_BORDER = "#DCE6F2"        # light card border
TRACK_BG = "#E7EFF8"            # progress bar track background


def rgba(hexcolor, alpha):
    c = QColor(hexcolor)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def make_shadow(blur=24, y_offset=6, alpha=18, color=NAVY_800):
    """Soft elevation shadow for cards sitting on a light background."""
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setOffset(0, y_offset)
    c = QColor(color)
    c.setAlpha(alpha)
    effect.setColor(c)
    return effect


class GlassPanel(QFrame):
    """White card panel with a soft border - the basic building block
    reused by every dashboard screen.

    NOTE: drop shadow removed by default (was QGraphicsDropShadowEffect on
    every card). On some Windows graphics drivers, many stacked/nested
    graphics effects can cause child widgets to fail to paint correctly.
    Call add_shadow() explicitly on a specific panel if you want the
    elevated look on select cards (e.g. just the header)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            QFrame{{
                background:{GLASS_FILL_STRONG};
                border:1px solid {GLASS_BORDER};
                border-radius:16px;
            }}
            QLabel{{ background:transparent; border:none; }}
        """)

    def add_shadow(self, **kwargs):
        self.setGraphicsEffect(make_shadow(**kwargs))
        return self