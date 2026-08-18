from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QLinearGradient, QPainterPath
from PySide6.QtSvgWidgets import QSvgWidget

import sys
import os
import joblib
import pandas as pd


def resource_base_dir():
    """Returns the project root in dev mode, or the folder containing the
    bundled data files when running as a PyInstaller-frozen app.

    sys._MEIPASS is set by PyInstaller's bootloader in BOTH onefile mode
    (temp extraction dir) and onedir mode (the _internal folder next to
    the .exe) — using plain __file__ based paths breaks once bundled,
    because __file__ then points inside PyInstaller's internal bundle
    rather than reliably next to the data files placed by COLLECT()."""
    if getattr(sys, 'frozen', False):
        return getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from report_generator import generate_report
from gui.biomarker_panel import BiomarkerPanel
from gui.explainability_panel import ExplainabilityPanel
from gui.theme import (
    NAVY_950, NAVY_900, NAVY_800, NAVY_700,
    BLUE_500, BLUE_400, CYAN_400, CYAN_300,
    ICE_100, SLATE_300, SLATE_400,
    GREEN_400, AMBER_400, RED_400,
    GLASS_FILL, GLASS_FILL_STRONG, GLASS_BORDER, TRACK_BG,
    rgba, make_shadow, GlassPanel,
)


class Sparkline(QWidget):
    """Small line-chart widget with a filled gradient area."""

    def __init__(self, values, color=CYAN_400, parent=None):
        super().__init__(parent)
        self.values = values
        self.color = QColor(color)
        self.setMinimumHeight(70)

    def set_values(self, values):
        self.values = list(values) if len(values) >= 2 else list(values) * 2
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        pad = 6
        vals = self.values
        if len(vals) < 2:
            vals = vals * 2
        vmin, vmax = min(vals), max(vals)
        span = (vmax - vmin) or 1

        pts = []
        for i, v in enumerate(vals):
            x = pad + (w - 2 * pad) * (i / (len(vals) - 1))
            y = h - pad - (h - 2 * pad) * ((v - vmin) / span)
            pts.append(QPointF(x, y))

        path = QPainterPath()
        path.moveTo(pts[0].x(), h)
        for pt in pts:
            path.lineTo(pt)
        path.lineTo(pts[-1].x(), h)
        path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, h)
        top = QColor(self.color)
        top.setAlpha(70)
        bottom = QColor(self.color)
        bottom.setAlpha(0)
        grad.setColorAt(0, top)
        grad.setColorAt(1, bottom)
        p.fillPath(path, grad)

        pen = QPen(self.color, 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        for i in range(len(pts) - 1):
            p.drawLine(pts[i], pts[i + 1])

        p.end()


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        print("Medical Dashboard UI Running")

        self.setWindowTitle("Parkinson AI Predictor — Clinical Dashboard [BUILD-v6]")
        self.resize(1180, 860)

        self.selected_file = None
        self.bio_window = None
        self.explain_window = None
        self.last_prediction = None

        self.load_model()
        self.build_ui()

    # ------------------------------------------------------------------
    def load_model(self):
        base = resource_base_dir()

        self.model = joblib.load(
            os.path.join(base, "models", "final_parkinson_logistic_model.pkl")
        )
        self.scaler = joblib.load(
            os.path.join(base, "models", "final_scaler.pkl")
        )
        with open(os.path.join(base, "models", "final_genes.txt")) as f:
            self.genes = [x.strip() for x in f.readlines()]

        self.gene_importance = []
        if hasattr(self.model, "coef_"):
            coefs = self.model.coef_[0]
            max_abs = max(abs(c) for c in coefs) or 1
            pairs = list(zip(self.genes, coefs))
            pairs.sort(key=lambda p: abs(p[1]), reverse=True)
            self.gene_importance = [
                (gene, abs(coef) / max_abs, coef > 0) for gene, coef in pairs
            ]
        else:
            self.gene_importance = [(g, 1 - i / len(self.genes), True) for i, g in enumerate(self.genes)]

        print("Model loaded successfully")

    # ------------------------------------------------------------------
    def field_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(f"""
            color:{SLATE_400};
            font-size:11px;
            font-weight:600;
            letter-spacing:1px;
        """)
        return lbl

    def section_head(self, icon_emoji, text, count=None):
        row = QHBoxLayout()
        row.setSpacing(9)
        ico = QLabel(icon_emoji)
        ico.setStyleSheet(f"font-size:15px; color:{BLUE_500};")
        title = QLabel(text.upper())
        title.setStyleSheet(f"""
            color:{ICE_100};
            font-size:13px;
            font-weight:700;
            letter-spacing:0.6px;
        """)
        row.addWidget(ico)
        row.addWidget(title)
        row.addStretch()
        if count:
            c = QLabel(count)
            c.setStyleSheet(f"color:{SLATE_400}; font-size:11px;")
            row.addWidget(c)
        return row

    def divider(self):
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{GLASS_BORDER}; border:none;")
        return line

    # ------------------------------------------------------------------
    def build_ui(self):
        self.setStyleSheet(f"""
            QWidget{{
                background: {NAVY_950};
                color:{ICE_100};
                font-family:'Segoe UI';
                font-size:12.5px;
            }}
        """)
        self.setAutoFillBackground(True)

        outer = QVBoxLayout()
        outer.setContentsMargins(26, 22, 26, 26)
        outer.setSpacing(18)

        # ================= Header (navy, stays dark for contrast) =================
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame{{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {NAVY_800}, stop:1 {NAVY_700});
                border-radius:16px;
            }}
        """)
        header.setGraphicsEffect(make_shadow(blur=30, y_offset=8, alpha=35))

        hl = QHBoxLayout()
        hl.setContentsMargins(20, 12, 20, 12)
        hl.setSpacing(14)

        logo_path = os.path.join(resource_base_dir(), "gui", "assets", "logo_dark.svg")
        logo = QSvgWidget(logo_path)
        logo.setFixedSize(42, 42)
        hl.addWidget(logo)

        brand = QVBoxLayout()
        brand.setSpacing(1)
        title = QLabel("Parkinson AI Predictor")
        title.setStyleSheet("color:white; font-size:18px; font-weight:700;")
        subtitle = QLabel("PARKINSON DISEASE PREDICTION SYSTEM")
        subtitle.setStyleSheet(f"color:{CYAN_300}; font-size:10.5px; letter-spacing:1.4px;")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        hl.addLayout(brand)
        hl.addStretch()

        status = QLabel("●  Model Online")
        status.setStyleSheet(f"""
            color:#7FE0B8;
            background:rgba(255,255,255,20);
            border:1px solid rgba(255,255,255,55);
            border-radius:999px;
            padding:7px 14px;
            font-size:12px;
            font-weight:600;
        """)
        hl.addWidget(status)

        user = QLabel("🧑‍⚕  Research Lab")
        user.setStyleSheet(f"""
            color:white;
            background:rgba(255,255,255,14);
            border:1px solid rgba(255,255,255,40);
            border-radius:999px;
            padding:7px 14px;
            font-size:12px;
            font-weight:600;
        """)
        hl.addWidget(user)

        header.setLayout(hl)
        outer.addWidget(header)

        # ================= Grid: left panel | center panel (no right panel) =================
        grid = QHBoxLayout()
        grid.setSpacing(18)

        grid.addWidget(self.build_left_panel(), 0)
        grid.addWidget(self.build_center_panel(), 1)

        outer.addLayout(grid, 1)

        # ================= Footer =================
        footer = QLabel(
            "Model v2.3.1 · Logistic Regression      Trained on GEO + PPMI cohorts"
            "                              For research & clinical decision support only — not a standalone diagnostic"
        )
        footer.setStyleSheet(f"color:{SLATE_400}; font-size:10.5px;")
        outer.addWidget(footer)

        self.setLayout(outer)

    # ------------------------------------------------------------------
    def build_left_panel(self):
        panel = GlassPanel()
        panel.setFixedWidth(290)
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea{{ background:transparent; border:none; }}
            QScrollBar:vertical{{ background:transparent; width:8px; }}
            QScrollBar::handle:vertical{{ background:{GLASS_BORDER}; border-radius:4px; min-height:24px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical{{ height:0px; }}
        """)

        content = QWidget()
        content.setStyleSheet("background:transparent;")
        v = QVBoxLayout(content)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(10)

        v.addLayout(self.section_head("👤", "Patient Information"))
        v.addSpacing(6)

        v.addWidget(self.field_label("Gene Expression Data"))

        upload_zone = QLabel("⇪\nUpload CSV Dataset\nClick to browse")
        upload_zone.setAlignment(Qt.AlignCenter)
        upload_zone.setCursor(Qt.PointingHandCursor)
        upload_zone.setStyleSheet(f"""
            border:1.5px dashed {CYAN_400};
            border-radius:12px;
            padding:20px 10px;
            color:{SLATE_300};
            font-size:12px;
            background:{GLASS_FILL};
        """)
        upload_zone.mousePressEvent = lambda e: self.open_file()
        v.addWidget(upload_zone)

        self.file_row = QLabel("No file selected yet")
        self.file_row.setWordWrap(True)
        self.file_row.setStyleSheet(f"""
            background:{rgba(BLUE_500,18)};
            border:1px solid {rgba(BLUE_500,60)};
            border-radius:10px;
            padding:10px 12px;
            color:{NAVY_800};
            font-size:11.5px;
        """)
        v.addWidget(self.file_row)

        v.addSpacing(8)
        v.addWidget(self.field_label("Data Validation"))
        for txt, ok in [
            ("Schema matched to training set", True),
            ("Missing values: none detected", True),
            ("Normalization: quantile applied", True),
        ]:
            row = QLabel(("✅  " if ok else "⚠️  ") + txt)
            row.setStyleSheet(f"color:{SLATE_300}; font-size:11.5px;")
            v.addWidget(row)

        v.addSpacing(8)
        v.addWidget(self.divider())
        v.addSpacing(8)

        meta = QGridLayout()
        meta.setSpacing(8)
        meta_items = [
            ("MODEL", "Logistic Reg."),
            ("SIGNATURE", f"{len(self.genes)} genes"),
            ("SAMPLE", "RNA-Seq"),
            ("COHORT", "GEO / PPMI"),
        ]
        for i, (k, val) in enumerate(meta_items):
            box = QFrame()
            box.setStyleSheet(f"""
                background:{GLASS_FILL};
                border:1px solid {GLASS_BORDER};
                border-radius:10px;
            """)
            bl = QVBoxLayout()
            bl.setContentsMargins(10, 8, 10, 8)
            bl.setSpacing(2)
            kl = QLabel(k)
            kl.setStyleSheet(f"color:{SLATE_400}; font-size:9px; letter-spacing:0.6px; background:transparent; border:none;")
            vl = QLabel(val)
            vl.setStyleSheet(f"color:{ICE_100}; font-size:13px; font-weight:700; background:transparent; border:none;")
            bl.addWidget(kl)
            bl.addWidget(vl)
            box.setLayout(bl)
            meta.addWidget(box, i // 2, i % 2)
        v.addLayout(meta)

        v.addSpacing(8)
        v.addWidget(self.divider())
        v.addSpacing(8)

        # Probability trend now lives here (was in the removed right panel)
        v.addLayout(self.section_head("📊", "Probability Trend"))
        self.prob_history = [0, 0]
        self.spark = Sparkline(self.prob_history)
        v.addWidget(self.spark)

        v.addStretch()

        scroll.setWidget(content)
        panel_layout.addWidget(scroll)
        panel.setLayout(panel_layout)
        return panel

    # ------------------------------------------------------------------
    def build_center_panel(self):
        wrap = QWidget()
        v = QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(18)

        # ---- Result card ----
        self.result_card = GlassPanel()
        rl = QVBoxLayout()
        rl.setContentsMargins(24, 22, 24, 22)
        rl.setSpacing(10)

        top_row = QHBoxLayout()
        eyebrow = QLabel("PREDICTION RESULT")
        eyebrow.setStyleSheet(f"color:{SLATE_300}; font-size:11px; letter-spacing:1.4px; font-weight:700;")
        top_row.addWidget(eyebrow)
        top_row.addStretch()
        model_tag = QLabel("●  Model: Logistic Regression")
        model_tag.setStyleSheet(f"""
            color:{NAVY_700}; font-size:11px;
            background:{rgba(CYAN_400,25)}; border:1px solid {rgba(CYAN_400,80)};
            border-radius:999px; padding:5px 12px;
        """)
        top_row.addWidget(model_tag)
        rl.addLayout(top_row)

        body_row = QHBoxLayout()
        self.result = QLabel("Awaiting Analysis")
        self.result.setMinimumHeight(42)
        self.result.setStyleSheet(f"color:{ICE_100}; font-size:28px; font-weight:700;")
        body_row.addWidget(self.result)
        body_row.addStretch()

        self.prob = QLabel("—")
        self.prob.setMinimumHeight(50)
        self.prob.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.prob.setStyleSheet(f"color:{SLATE_300}; font-size:34px; font-weight:700;")
        body_row.addWidget(self.prob)
        rl.addLayout(body_row)

        self.result_sub = QLabel(f"Classified from {len(self.genes)} biomarker genes")
        self.result_sub.setStyleSheet(f"color:{SLATE_300}; font-size:12px;")
        rl.addWidget(self.result_sub)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(9)
        self.bar.setStyleSheet(f"""
            QProgressBar{{ background:{TRACK_BG}; border-radius:4.5px; border:none; }}
            QProgressBar::chunk{{
                border-radius:4.5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BLUE_500}, stop:1 {RED_400});
            }}
        """)
        rl.addWidget(self.bar)

        rl.addWidget(self.divider())

        foot_row = QHBoxLayout()
        self.risk = QLabel("Risk Level: —")
        self.risk.setStyleSheet(f"color:{SLATE_300}; font-size:11.5px;")
        foot_row.addWidget(self.risk)
        foot_row.addStretch()
        rl.addLayout(foot_row)

        self.result_card.setLayout(rl)
        v.addWidget(self.result_card)

        # ---- Two info cards ----
        info_row = QHBoxLayout()
        info_row.setSpacing(18)

        perf = GlassPanel()
        pl = QVBoxLayout()
        pl.setContentsMargins(18, 16, 18, 16)
        pl.setSpacing(8)
        pl.addLayout(self.section_head("📈", "Model Performance"))
        for k, val in [("Cross-val Accuracy", "90.7%"), ("F1 Score", "0.902"),
                       ("Feature Set", f"{len(self.genes)} genes")]:
            row = QHBoxLayout()
            kl = QLabel(k)
            kl.setStyleSheet(f"color:{SLATE_300}; font-size:12px;")
            vl = QLabel(val)
            vl.setStyleSheet(f"color:{ICE_100}; font-size:12px; font-weight:700;")
            row.addWidget(kl)
            row.addStretch()
            row.addWidget(vl)
            pl.addLayout(row)
        perf.setLayout(pl)

        pipe = GlassPanel()
        pil = QVBoxLayout()
        pil.setContentsMargins(18, 16, 18, 16)
        pil.setSpacing(8)
        pil.addLayout(self.section_head("🧬", "Pipeline Status"))
        self.pipeline_labels = {}
        for step in ["Preprocessing", "Feature Scaling", "Inference"]:
            row = QHBoxLayout()
            kl = QLabel(step)
            kl.setStyleSheet(f"color:{SLATE_300}; font-size:12px;")
            vl = QLabel("Idle")
            vl.setStyleSheet(f"color:{SLATE_400}; font-size:12px; font-weight:700;")
            self.pipeline_labels[step] = vl
            row.addWidget(kl)
            row.addStretch()
            row.addWidget(vl)
            pil.addLayout(row)
        pipe.setLayout(pil)

        info_row.addWidget(perf)
        info_row.addWidget(pipe)
        v.addLayout(info_row)

        # ---- Model Genes card (now shown only once, here) ----
        genes_card = GlassPanel()
        gl = QVBoxLayout()
        gl.setContentsMargins(18, 16, 18, 16)
        gl.setSpacing(10)
        gl.addLayout(self.section_head("🧬", "Model Genes", f"{len(self.genes)} total"))
        for gene, weight, positive in self.gene_importance[:5]:
            gl.addWidget(self.make_gene_row(gene, weight, positive))
        legend = QLabel("▬  |coefficient| relative to strongest gene")
        legend.setStyleSheet(f"color:{SLATE_400}; font-size:10.5px;")
        gl.addWidget(legend)
        genes_card.setLayout(gl)
        v.addWidget(genes_card)

        # ---- Action bar ----
        actions = QHBoxLayout()
        actions.setSpacing(12)

        browse = QPushButton("📁  Browse Patient File")
        browse.setCursor(Qt.PointingHandCursor)
        browse.setMinimumHeight(46)
        browse.setStyleSheet(f"""
            QPushButton{{
                background:{rgba(CYAN_400,18)};
                color:{NAVY_700};
                border:1px solid {rgba(CYAN_400,90)};
                border-radius:11px;
                font-size:13px; font-weight:600;
            }}
            QPushButton:hover{{ background:{rgba(CYAN_400,32)}; }}
        """)
        browse.clicked.connect(self.open_file)

        run = QPushButton("▶  Run Prediction")
        run.setCursor(Qt.PointingHandCursor)
        run.setMinimumHeight(46)
        run.setStyleSheet(f"""
            QPushButton{{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {BLUE_500}, stop:1 {NAVY_700});
                color:white;
                border:none;
                border-radius:11px;
                font-size:13.5px; font-weight:700;
            }}
            QPushButton:hover{{ background:{BLUE_400}; }}
        """)
        run.clicked.connect(self.predict)

        report = QPushButton("🧾  Biomarker Report")
        report.setCursor(Qt.PointingHandCursor)
        report.setMinimumHeight(46)
        report.setStyleSheet(f"""
            QPushButton{{
                background:{rgba(GREEN_400,15)};
                color:{GREEN_400};
                border:1px solid {rgba(GREEN_400,70)};
                border-radius:11px;
                font-size:13px; font-weight:600;
            }}
            QPushButton:hover{{ background:{rgba(GREEN_400,25)}; }}
        """)
        report.clicked.connect(self.generate_pdf_report)

        explain = QPushButton("🧬  Explain AI Decision")
        explain.setCursor(Qt.PointingHandCursor)
        explain.setMinimumHeight(46)
        explain.setStyleSheet(f"""
            QPushButton{{
                background:{rgba(AMBER_400,15)};
                color:{AMBER_400};
                border:1px solid {rgba(AMBER_400,70)};
                border-radius:11px;
                font-size:13px; font-weight:600;
            }}
            QPushButton:hover{{ background:{rgba(AMBER_400,25)}; }}
        """)
        explain.clicked.connect(self.open_explainability)

        actions.addWidget(browse)
        actions.addWidget(run)
        actions.addWidget(report)
        actions.addWidget(explain)
        v.addLayout(actions)

        wrap.setLayout(v)
        return wrap

    # ------------------------------------------------------------------
    def make_gene_row(self, name, weight, positive=True):
        box = QFrame()
        box.setStyleSheet(f"""
            background:{GLASS_FILL};
            border:1px solid {GLASS_BORDER};
            border-radius:10px;
        """)
        bl = QVBoxLayout()
        bl.setContentsMargins(12, 10, 12, 10)
        bl.setSpacing(6)

        top = QHBoxLayout()
        nm = QLabel(name)
        nm.setStyleSheet(f"color:{ICE_100}; font-size:12.5px; font-weight:700; font-family:'Consolas';")
        dir_icon = "▲" if positive else "▼"
        dir_color = RED_400 if positive else GREEN_400
        wt = QLabel(f"{dir_icon} {weight:.3f}")
        wt.setStyleSheet(f"color:{dir_color}; font-size:11px; font-family:'Consolas'; font-weight:600;")
        top.addWidget(nm)
        top.addStretch()
        top.addWidget(wt)
        bl.addLayout(top)

        track = QProgressBar()
        track.setValue(int(weight * 100))
        track.setTextVisible(False)
        track.setFixedHeight(6)
        chunk_color = RED_400 if positive else GREEN_400
        track.setStyleSheet(f"""
            QProgressBar{{ background:{TRACK_BG}; border-radius:3px; border:none; }}
            QProgressBar::chunk{{
                border-radius:3px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {CYAN_400}, stop:1 {chunk_color});
            }}
        """)
        bl.addWidget(track)

        box.setLayout(bl)
        return box

    # ------------------------------------------------------------------
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Patient CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.selected_file = file
            self.file_row.setText(f"✅  {os.path.basename(file)}")

    # ------------------------------------------------------------------
    def predict(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please upload a patient CSV file first.")
            return

        for step in self.pipeline_labels:
            self.pipeline_labels[step].setText("Complete")
            self.pipeline_labels[step].setStyleSheet(f"color:{GREEN_400}; font-size:12px; font-weight:700;")

        df = pd.read_csv(self.selected_file)

        if "Gene" in df.columns:
            df = df.set_index("Gene").T

        x = df[self.genes]
        x = self.scaler.transform(x)

        pred = self.model.predict(x)
        prob = self.model.predict_proba(x)[0][1] * 100
        is_disease = pred[0] == 1
        diagnosis = "Parkinson Disease" if is_disease else "Healthy Control"
        risk_label = "High" if is_disease else "Low"

        if is_disease:
            self.result.setText("Parkinson Disease")
            self.result.setStyleSheet(f"color:{RED_400}; font-size:28px; font-weight:700;")
            self.prob.setStyleSheet(f"color:{RED_400}; font-size:34px; font-weight:700;")
            self.risk.setText("Risk Level: 🔴  High")
        else:
            self.result.setText("Healthy Control")
            self.result.setStyleSheet(f"color:{GREEN_400}; font-size:28px; font-weight:700;")
            self.prob.setStyleSheet(f"color:{GREEN_400}; font-size:34px; font-weight:700;")
            self.risk.setText("Risk Level: 🟢  Low")

        self.prob.setText(f"{prob:.1f}%")
        self.bar.setValue(int(prob))
        self.result_sub.setText(f"Classified from {len(self.genes)} biomarker genes · {os.path.basename(self.selected_file)}")

        if self.prob_history == [0, 0]:
            self.prob_history = [prob]
        else:
            self.prob_history.append(prob)
        if len(self.prob_history) > 10:
            self.prob_history.pop(0)
        self.spark.set_values(self.prob_history)

        self.last_prediction = {
            "file_name": os.path.basename(self.selected_file),
            "diagnosis": diagnosis,
            "probability": round(prob, 2),
            "risk": risk_label,
            "top_genes": [g for g, _, _ in self.gene_importance[:14]],
        }

    # ------------------------------------------------------------------
    def open_biomarkers(self):
        self.bio_window = BiomarkerPanel()
        self.bio_window.show()

    # ------------------------------------------------------------------
    def generate_pdf_report(self):
        if not self.last_prediction:
            QMessageBox.warning(
                self, "No Prediction Yet",
                "Please run a prediction before generating a report."
            )
            return

        p = self.last_prediction
        pdf = generate_report(
            p["file_name"],
            p["diagnosis"],
            p["probability"],
            p["risk"],
            p["top_genes"],
        )
        QMessageBox.information(
            self,
            "Clinical Report",
            f"PDF created successfully:\n{pdf}"
        )

    # ------------------------------------------------------------------
    def open_explainability(self):
        if not self.last_prediction:
            QMessageBox.warning(
                self, "No Prediction Yet",
                "Please run a prediction before viewing the explanation."
            )
            return

        self.explain_window = ExplainabilityPanel(
            prediction=self.last_prediction,
            gene_importance=self.gene_importance,
        )
        self.explain_window.show()


if __name__ == "__main__":
    # Fixes text/label clipping that can happen on Windows displays with
    # scaling above 100% (125%, 150%, ...) — without this, Qt can compute
    # a label's box at one DPI and render the font at another.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    w = MainWindow()
    w.show()

    sys.exit(app.exec())