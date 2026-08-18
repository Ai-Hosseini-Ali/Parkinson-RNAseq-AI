from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

import sys
import os
import joblib
import pandas as pd

from gui.biomarker_panel import BiomarkerPanel


# ---------------------------------------------------------------------------
# Clinical color palette
# ---------------------------------------------------------------------------
COLOR_PRIMARY_DARK = "#0B2545"      # deep navy (header)
COLOR_PRIMARY = "#134074"           # medical navy blue
COLOR_ACCENT = "#0F8B8D"            # clinical teal
COLOR_ACCENT_HOVER = "#0C6E70"
COLOR_BG = "#EEF3F8"                # cool clinical background
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#DCE6F0"
COLOR_TEXT = "#1B2A41"
COLOR_MUTED = "#5C7089"
COLOR_DANGER = "#C0392B"
COLOR_SAFE = "#1E8A5F"


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        print("Medical Dashboard UI Running")

        self.setWindowTitle("Parkinson AI Predictor — Clinical Decision Support")
        self.resize(1150, 780)
        self.setStyleSheet(f"background:{COLOR_BG};")

        self.selected_file = None
        self.bio_window = None

        self.load_model()
        self.build_ui()

    # ------------------------------------------------------------------
    def load_model(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.model = joblib.load(
            os.path.join(base, "models", "final_parkinson_logistic_model.pkl")
        )
        self.scaler = joblib.load(
            os.path.join(base, "models", "final_scaler.pkl")
        )
        with open(os.path.join(base, "models", "final_genes.txt")) as f:
            self.genes = [x.strip() for x in f.readlines()]

        print("Model loaded successfully")

    # ------------------------------------------------------------------
    def card_style(self):
        return f"""
        QFrame{{
            background:{COLOR_CARD};
            border-radius:14px;
            border:1px solid {COLOR_BORDER};
        }}
        QLabel{{
            color:{COLOR_TEXT};
            border:none;
        }}
        QPushButton{{
            background:{COLOR_ACCENT};
            color:white;
            padding:11px 18px;
            border-radius:8px;
            font-size:14px;
            font-weight:600;
            border:none;
        }}
        QPushButton:hover{{
            background:{COLOR_ACCENT_HOVER};
        }}
        QPushButton:pressed{{
            background:{COLOR_PRIMARY};
        }}
        """

    def section_title_style(self):
        return f"""
            font-size:13px;
            font-weight:700;
            color:{COLOR_MUTED};
            letter-spacing:1px;
            text-transform:uppercase;
            padding-bottom:4px;
            border-bottom:1px solid {COLOR_BORDER};
        """

    # ------------------------------------------------------------------
    def build_ui(self):
        main = QVBoxLayout()
        main.setContentsMargins(28, 24, 28, 24)
        main.setSpacing(18)

        # ---------------- Header ----------------
        header = QFrame()
        header.setStyleSheet(f"""
            background:{COLOR_PRIMARY_DARK};
            border-radius:16px;
        """)
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(30, 22, 30, 22)
        header_layout.setSpacing(4)

        title = QLabel("Parkinson AI Predictor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color:white;
            font-size:26px;
            font-weight:800;
            border:none;
        """)

        subtitle = QLabel("RNA-Seq Biomarker–Based Clinical Decision Support System")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"""
            color:#BFD7EA;
            font-size:13px;
            font-weight:500;
            letter-spacing:0.5px;
            border:none;
        """)

        badge = QLabel("● AI MODEL ACTIVE   |   14-GENE SIGNATURE   |   LOGISTIC REGRESSION")
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"""
            color:{COLOR_ACCENT};
            font-size:11px;
            font-weight:700;
            letter-spacing:1px;
            border:none;
            padding-top:6px;
        """)

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(badge)
        header.setLayout(header_layout)
        main.addWidget(header)

        # ---------------- Middle: Patient + Result cards ----------------
        middle = QHBoxLayout()
        middle.setSpacing(18)

        # --- Patient card ---
        patient = QFrame()
        patient.setStyleSheet(self.card_style())
        pl = QVBoxLayout()
        pl.setContentsMargins(22, 20, 22, 22)
        pl.setSpacing(14)

        p_title = QLabel("PATIENT DATA INPUT")
        p_title.setStyleSheet(self.section_title_style())
        pl.addWidget(p_title)

        p_desc = QLabel("Upload RNA-Seq expression profile (CSV format)\ncontaining the 14 required biomarker genes.")
        p_desc.setStyleSheet(f"color:{COLOR_MUTED}; font-size:12px; border:none;")
        pl.addWidget(p_desc)

        self.file_label = QLabel("⛶  No file selected")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"""
            background:{COLOR_BG};
            border:1px dashed {COLOR_BORDER};
            border-radius:8px;
            padding:14px;
            color:{COLOR_MUTED};
            font-size:12px;
        """)
        pl.addWidget(self.file_label)

        upload = QPushButton("📁  Upload Patient CSV")
        upload.setCursor(Qt.PointingHandCursor)
        upload.clicked.connect(self.open_file)
        pl.addWidget(upload)

        pl.addStretch()
        patient.setLayout(pl)

        # --- Result card ---
        result = QFrame()
        result.setStyleSheet(self.card_style())
        rl = QVBoxLayout()
        rl.setContentsMargins(22, 20, 22, 22)
        rl.setSpacing(12)

        r_title = QLabel("AI DIAGNOSTIC OUTPUT")
        r_title.setStyleSheet(self.section_title_style())
        rl.addWidget(r_title)

        self.result = QLabel("Awaiting Analysis")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet(f"""
            font-size:24px;
            font-weight:800;
            color:{COLOR_PRIMARY};
            padding:10px 0 2px 0;
            border:none;
        """)
        rl.addWidget(self.result)

        self.prob = QLabel("Probability: —")
        self.prob.setAlignment(Qt.AlignCenter)
        self.prob.setStyleSheet(f"color:{COLOR_MUTED}; font-size:13px; border:none;")
        rl.addWidget(self.prob)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(f"""
            QProgressBar{{
                background:{COLOR_BG};
                border-radius:5px;
                border:none;
            }}
            QProgressBar::chunk{{
                background:{COLOR_ACCENT};
                border-radius:5px;
            }}
        """)
        rl.addWidget(self.bar)

        self.risk = QLabel("Risk Level: Unknown")
        self.risk.setAlignment(Qt.AlignCenter)
        self.risk.setStyleSheet(f"""
            font-size:13px;
            font-weight:700;
            color:{COLOR_MUTED};
            background:{COLOR_BG};
            border-radius:8px;
            padding:10px;
            border:none;
        """)
        rl.addWidget(self.risk)

        rl.addStretch()
        result.setLayout(rl)

        middle.addWidget(patient, 1)
        middle.addWidget(result, 1)
        main.addLayout(middle)

        # ---------------- Action buttons ----------------
        buttons = QHBoxLayout()
        buttons.setSpacing(14)

        predict = QPushButton("⚕  Analyze Patient")
        predict.setCursor(Qt.PointingHandCursor)
        predict.setStyleSheet(f"""
            QPushButton{{
                background:{COLOR_PRIMARY};
                color:white;
                padding:14px;
                border-radius:10px;
                font-size:15px;
                font-weight:700;
                border:none;
            }}
            QPushButton:hover{{ background:{COLOR_PRIMARY_DARK}; }}
        """)
        predict.clicked.connect(self.predict)

        biomarkers = QPushButton("🧬  Biomarker Report")
        biomarkers.setCursor(Qt.PointingHandCursor)
        biomarkers.setStyleSheet(f"""
            QPushButton{{
                background:{COLOR_CARD};
                color:{COLOR_PRIMARY};
                padding:14px;
                border-radius:10px;
                font-size:15px;
                font-weight:700;
                border:1px solid {COLOR_BORDER};
            }}
            QPushButton:hover{{ background:{COLOR_BG}; }}
        """)
        biomarkers.clicked.connect(self.open_biomarkers)

        buttons.addWidget(predict)
        buttons.addWidget(biomarkers)
        main.addLayout(buttons)

        # ---------------- Footer ----------------
        footer = QFrame()
        footer.setStyleSheet(f"""
            background:transparent;
            border-top:1px solid {COLOR_BORDER};
        """)
        fl = QHBoxLayout()
        fl.setContentsMargins(4, 12, 4, 0)

        footer_text = QLabel(
            "Model: Logistic Regression   •   Signature: 14 Biomarker Genes   •   "
            "For research and clinical decision-support use only"
        )
        footer_text.setAlignment(Qt.AlignCenter)
        footer_text.setStyleSheet(f"color:{COLOR_MUTED}; font-size:11px; border:none;")
        fl.addWidget(footer_text)
        footer.setLayout(fl)
        main.addWidget(footer)

        self.setLayout(main)

    # ------------------------------------------------------------------
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select Patient CSV", "", "CSV Files (*.csv)"
        )
        if file:
            self.selected_file = file
            self.file_label.setText(f"✅  {os.path.basename(file)}")

    # ------------------------------------------------------------------
    def predict(self):
        if not self.selected_file:
            QMessageBox.warning(self, "No File", "Please upload a patient CSV file first.")
            return

        df = pd.read_csv(self.selected_file)

        if "Gene" in df.columns:
            df = df.set_index("Gene").T

        x = df[self.genes]
        x = self.scaler.transform(x)

        pred = self.model.predict(x)
        prob = self.model.predict_proba(x)[0][1] * 100

        if pred[0] == 1:
            self.result.setText("Parkinson's Disease Detected")
            self.result.setStyleSheet(f"""
                font-size:22px; font-weight:800; color:{COLOR_DANGER}; border:none;
            """)
        else:
            self.result.setText("Healthy Control")
            self.result.setStyleSheet(f"""
                font-size:22px; font-weight:800; color:{COLOR_SAFE}; border:none;
            """)

        self.prob.setText(f"Probability: {prob:.2f}%")
        self.bar.setValue(int(prob))

        if prob > 70:
            self.risk.setText("🔴  High Risk")
            self.risk.setStyleSheet(f"""
                font-size:13px; font-weight:700; color:{COLOR_DANGER};
                background:#FDECEA; border-radius:8px; padding:10px; border:none;
            """)
        else:
            self.risk.setText("🟢  Low Risk")
            self.risk.setStyleSheet(f"""
                font-size:13px; font-weight:700; color:{COLOR_SAFE};
                background:#E9F7EF; border-radius:8px; padding:10px; border:none;
            """)

    # ------------------------------------------------------------------
    def open_biomarkers(self):
        self.bio_window = BiomarkerPanel()
        self.bio_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())