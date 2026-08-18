"""
AI Decision Explanation panel — turns the raw prediction into something a
clinician can actually read: which genes drove the decision, how confident
the model is, and a plain-language clinical summary.

Everything here is populated from the *real* prediction (self.last_prediction
and self.gene_importance in main_window.py) — nothing is hardcoded demo data.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont

from gui.theme import (
    NAVY_950, NAVY_700, ICE_100, SLATE_300, SLATE_400,
    CYAN_400, CYAN_300, GREEN_400, AMBER_400, RED_400,
    GLASS_BORDER, GLASS_FILL, TRACK_BG, rgba, GlassPanel,
)


class CircularGauge(QWidget):
    """Circular confidence/probability gauge, drawn with QPainter since Qt
    has no built-in circular progress widget."""

    def __init__(self, value=0, color=CYAN_400, parent=None):
        super().__init__(parent)
        self.value = value  # 0-100
        self.color = QColor(color)
        self.setFixedSize(130, 130)

    def set_value(self, value, color=None):
        self.value = value
        if color:
            self.color = QColor(color)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        thickness = 11

        track_pen = QPen(QColor(TRACK_BG), thickness)
        track_pen.setCapStyle(Qt.RoundCap)
        p.setPen(track_pen)
        p.drawArc(rect, 90 * 16, -360 * 16)

        val_pen = QPen(self.color, thickness)
        val_pen.setCapStyle(Qt.RoundCap)
        p.setPen(val_pen)
        span = -int(360 * 16 * (self.value / 100))
        p.drawArc(rect, 90 * 16, span)

        p.setPen(QColor(ICE_100))
        font = QFont("Segoe UI", 20, QFont.Bold)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, f"{self.value:.1f}%")

        p.end()


class ExplainabilityPanel(QWidget):

    def __init__(self, prediction=None, gene_importance=None, parent=None):
        super().__init__(parent)

        self.prediction = prediction or {}
        self.gene_importance = gene_importance or []

        self.setWindowTitle("AI Decision Explanation")
        self.resize(760, 860)
        self.setStyleSheet(f"""
            QWidget{{
                background:{NAVY_950};
                color:{ICE_100};
                font-family:'Segoe UI';
                font-size:12.5px;
            }}
        """)

        self.build_ui()

    # ------------------------------------------------------------------
    def section_head(self, icon_emoji, text):
        row = QHBoxLayout()
        row.setSpacing(9)
        ico = QLabel(icon_emoji)
        ico.setStyleSheet(f"font-size:15px; color:{NAVY_700};")
        title = QLabel(text.upper())
        title.setStyleSheet(f"""
            color:{ICE_100}; font-size:13px; font-weight:700; letter-spacing:0.6px;
        """)
        row.addWidget(ico)
        row.addWidget(title)
        row.addStretch()
        return row

    # ------------------------------------------------------------------
    def build_ui(self):
        page = QVBoxLayout(self)
        page.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea{{ border:none; background:{NAVY_950}; }}
            QScrollBar:vertical{{ background:transparent; width:10px; }}
            QScrollBar::handle:vertical{{ background:{GLASS_BORDER}; border-radius:5px; min-height:30px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{{ height:0px; }}
        """)

        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(18)

        # ================= Header =================
        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("🧬  AI Decision Explanation")
        title.setStyleSheet(f"color:{ICE_100}; font-size:21px; font-weight:800;")
        subtitle = QLabel("MACHINE LEARNING INTERPRETATION MODULE")
        subtitle.setStyleSheet(f"color:{SLATE_300}; font-size:11px; letter-spacing:1.6px;")
        header.addWidget(title)
        header.addWidget(subtitle)
        outer.addLayout(header)

        outer.addWidget(self.build_prediction_card())
        outer.addWidget(self.build_gene_contribution_card())
        outer.addWidget(self.build_clinical_summary_card())
        outer.addWidget(self.build_model_info_card())

        outer.addStretch()
        scroll.setWidget(content)
        page.addWidget(scroll)

    # ------------------------------------------------------------------
    def build_prediction_card(self):
        card = GlassPanel()
        row = QHBoxLayout()
        row.setContentsMargins(24, 22, 24, 22)
        row.setSpacing(24)

        diagnosis = self.prediction.get("diagnosis", "—")
        prob = self.prediction.get("probability", 0.0)
        is_disease = diagnosis.lower().startswith("parkinson")
        main_color = RED_400 if is_disease else GREEN_400

        left = QVBoxLayout()
        left.setSpacing(6)

        eyebrow = QLabel("MODEL PREDICTION")
        eyebrow.setStyleSheet(f"color:{SLATE_300}; font-size:11px; letter-spacing:1.4px; font-weight:700;")
        left.addWidget(eyebrow)

        headline = QLabel(f"{diagnosis} Risk" if is_disease else diagnosis)
        headline.setStyleSheet(f"color:{ICE_100}; font-size:22px; font-weight:800;")
        headline.setWordWrap(True)
        left.addWidget(headline)

        left.addSpacing(6)

        conf_pct = abs(prob - 50) * 2
        if conf_pct >= 70:
            conf_label, conf_color = "High Confidence", GREEN_400
        elif conf_pct >= 40:
            conf_label, conf_color = "Moderate Confidence", AMBER_400
        else:
            conf_label, conf_color = "Low Confidence", RED_400

        conf_row = QHBoxLayout()
        conf_caption = QLabel("Confidence Score")
        conf_caption.setStyleSheet(f"color:{SLATE_300}; font-size:12px;")
        conf_row.addWidget(conf_caption)
        conf_row.addStretch()
        conf_val = QLabel(conf_label)
        conf_val.setStyleSheet(f"""
            color:{conf_color}; font-size:12px; font-weight:700;
            background:{rgba(conf_color, 22)};
            border:1px solid {rgba(conf_color, 80)}; border-radius:999px; padding:4px 12px;
        """)
        conf_row.addWidget(conf_val)
        left.addLayout(conf_row)

        conf_bar = QProgressBar()
        conf_bar.setValue(int(conf_pct))
        conf_bar.setTextVisible(False)
        conf_bar.setFixedHeight(8)
        conf_bar.setStyleSheet(f"""
            QProgressBar{{ background:{TRACK_BG}; border-radius:4px; border:none; }}
            QProgressBar::chunk{{ border-radius:4px; background:{conf_color}; }}
        """)
        left.addWidget(conf_bar)

        left.addStretch()
        row.addLayout(left, 1)

        self.gauge = CircularGauge(prob, main_color)
        gauge_wrap = QVBoxLayout()
        gauge_wrap.addWidget(self.gauge, 0, Qt.AlignCenter)
        gauge_caption = QLabel("Parkinson Probability")
        gauge_caption.setAlignment(Qt.AlignCenter)
        gauge_caption.setStyleSheet(f"color:{SLATE_400}; font-size:10.5px;")
        gauge_wrap.addWidget(gauge_caption)
        row.addLayout(gauge_wrap)

        card.setLayout(row)
        return card

    # ------------------------------------------------------------------
    def build_gene_contribution_card(self):
        card = GlassPanel()
        v = QVBoxLayout()
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(12)

        v.addLayout(self.section_head("🧬", "Top Contributing Biomarkers"))

        top_genes = self.gene_importance[:5] if self.gene_importance else []
        if not top_genes:
            empty = QLabel("No gene importance data available.")
            empty.setStyleSheet(f"color:{SLATE_400}; font-size:12px;")
            v.addWidget(empty)
        else:
            for gene, weight, positive in top_genes:
                v.addWidget(self.make_contribution_row(gene, weight, positive))

        card.setLayout(v)
        return card

    def make_contribution_row(self, name, weight, positive):
        box = QFrame()
        box.setStyleSheet(f"""
            background:{GLASS_FILL};
            border:1px solid {GLASS_BORDER};
            border-radius:10px;
        """)
        v = QVBoxLayout()
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(6)

        top = QHBoxLayout()
        nm = QLabel(name)
        nm.setStyleSheet(f"color:{ICE_100}; font-size:13px; font-weight:700; font-family:'Consolas';")
        pct = QLabel(f"{weight * 100:.0f}%")
        pct.setStyleSheet(f"color:{NAVY_700}; font-size:12px; font-weight:700; font-family:'Consolas';")
        top.addWidget(nm)
        top.addStretch()
        top.addWidget(pct)
        v.addLayout(top)

        bar_color = RED_400 if positive else GREEN_400
        bar = QProgressBar()
        bar.setValue(int(weight * 100))
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        bar.setStyleSheet(f"""
            QProgressBar{{ background:{TRACK_BG}; border-radius:4px; border:none; }}
            QProgressBar::chunk{{
                border-radius:4px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {CYAN_400}, stop:1 {bar_color});
            }}
        """)
        v.addWidget(bar)

        direction_text = "Positive contribution (↑ disease risk)" if positive else "Protective contribution (↓ disease risk)"
        direction = QLabel(direction_text)
        direction.setStyleSheet(f"color:{bar_color}; font-size:10.5px; font-weight:600;")
        v.addWidget(direction)

        box.setLayout(v)
        return box

    # ------------------------------------------------------------------
    def build_clinical_summary_card(self):
        """Clinical AI Summary: intro + bulleted main drivers + interpretation.
        Phrasing stays to statistical contribution direction only — it never
        invents biological mechanisms the model has no way of knowing."""
        card = GlassPanel()
        v = QVBoxLayout()
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(10)

        v.addLayout(self.section_head("🩺", "Clinical AI Summary"))

        diagnosis = self.prediction.get("diagnosis", "")
        prob = self.prediction.get("probability", 0.0)
        is_disease = diagnosis.lower().startswith("parkinson")

        if is_disease:
            intro = (
                "The model identified a gene expression pattern that closely "
                "resembles the Parkinson's disease profile it learned during training."
            )
        elif diagnosis:
            intro = (
                "The model did not identify a gene expression pattern consistent "
                "with Parkinson's disease for this sample."
            )
        else:
            intro = "Run a prediction to generate a clinical summary."
        intro_lbl = QLabel(intro)
        intro_lbl.setWordWrap(True)
        intro_lbl.setStyleSheet(f"color:{SLATE_300}; font-size:12.5px;")
        v.addWidget(intro_lbl)

        if self.gene_importance:
            drivers_title = QLabel("Main drivers:")
            drivers_title.setStyleSheet(f"color:{ICE_100}; font-size:12px; font-weight:700; padding-top:4px;")
            v.addWidget(drivers_title)

            for gene, weight, positive in self.gene_importance[:3]:
                phrase = "increased risk contribution" if positive else "protective influence"
                color = RED_400 if positive else GREEN_400
                line = QLabel(f"•  <b>{gene}</b> — {phrase} ({weight*100:.0f}% relative weight)")
                line.setTextFormat(Qt.RichText)
                line.setStyleSheet(f"color:{color}; font-size:12px; padding-left:4px;")
                v.addWidget(line)

        interp_title = QLabel("Interpretation:")
        interp_title.setStyleSheet(f"color:{ICE_100}; font-size:12px; font-weight:700; padding-top:6px;")
        v.addWidget(interp_title)

        if diagnosis:
            if is_disease:
                strength = "strong" if prob >= 80 else "moderate"
                interp = (
                    f"{strength.capitalize()} molecular similarity with the Parkinson "
                    f"disease profile ({prob:.1f}% probability)."
                )
            else:
                interp = (
                    f"The overall biomarker profile is more consistent with the "
                    f"healthy-control class ({100 - prob:.1f}% probability of healthy)."
                )
        else:
            interp = "—"
        interp_lbl = QLabel(interp)
        interp_lbl.setWordWrap(True)
        interp_lbl.setStyleSheet(f"color:{SLATE_300}; font-size:12.5px;")
        v.addWidget(interp_lbl)

        disclaimer = QLabel(
            "This summary reflects statistical feature contributions from a "
            "logistic regression model. It is a decision-support aid, not a "
            "standalone diagnosis."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(f"color:{SLATE_400}; font-size:10.5px; font-style:italic; padding-top:6px;")
        v.addWidget(disclaimer)

        card.setLayout(v)
        return card

    # ------------------------------------------------------------------
    def build_model_info_card(self):
        card = GlassPanel()
        v = QVBoxLayout()
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(10)

        v.addLayout(self.section_head("⚙", "Model Information"))

        n_genes = len(self.gene_importance) if self.gene_importance else 0

        # NOTE: "Training Dataset" and "Validation" reflect the general
        # methodology described in this project (GEO + PPMI cohorts).
        # Edit these two values if your actual training/validation setup
        # differs — this file cannot verify those claims on its own.
        rows = [
            ("Algorithm", "Logistic Regression"),
            ("Features", f"{n_genes} Selected Biomarkers"),
            ("Training Dataset", "GEO + PPMI cohorts"),
            ("Validation", "Internal cross-validation"),
        ]

        grid = QVBoxLayout()
        grid.setSpacing(8)
        for k, val in rows:
            row = QHBoxLayout()
            kl = QLabel(k)
            kl.setStyleSheet(f"color:{SLATE_300}; font-size:12px;")
            vl = QLabel(val)
            vl.setStyleSheet(f"color:{ICE_100}; font-size:12px; font-weight:700;")
            row.addWidget(kl)
            row.addStretch()
            row.addWidget(vl)
            grid.addLayout(row)
        v.addLayout(grid)

        card.setLayout(v)
        return card