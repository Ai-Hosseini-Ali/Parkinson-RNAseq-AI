from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QMessageBox,
    QProgressBar,
    QFrame,
)

import sys
import os
import joblib
import pandas as pd

from gui.biomarker_panel import BiomarkerPanel



class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Parkinson AI Predictor"
        )

        self.resize(
            750,
            700
        )


        self.selected_file = None

        self.bio_window = None


        self.load_model()

        self.build_ui()



    def load_model(self):

        try:

            base_path = os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )
            )


            self.model = joblib.load(
                os.path.join(
                    base_path,
                    "models",
                    "final_parkinson_logistic_model.pkl"
                )
            )


            self.scaler = joblib.load(
                os.path.join(
                    base_path,
                    "models",
                    "final_scaler.pkl"
                )
            )


            with open(
                os.path.join(
                    base_path,
                    "models",
                    "final_genes.txt"
                ),
                "r"
            ) as f:

                self.genes = [
                    g.strip()
                    for g in f.readlines()
                ]


            print(
                "Model loaded successfully"
            )


        except Exception as e:

            QMessageBox.critical(
                self,
                "Model Error",
                str(e)
            )



    def build_ui(self):

        layout = QVBoxLayout()



        title = QLabel(
            "🧠 Parkinson AI Predictor\n"
            "Parkinson Disease Prediction System"
        )


        title.setStyleSheet(
            """
            font-size:26px;
            font-weight:bold;
            """
        )


        layout.addWidget(
            title
        )



        self.file_label = QLabel(
            "No patient selected"
        )


        layout.addWidget(
            self.file_label
        )



        browse_btn = QPushButton(
            "📂 Browse Patient CSV"
        )


        browse_btn.clicked.connect(
            self.open_file
        )


        layout.addWidget(
            browse_btn
        )



        predict_btn = QPushButton(
            "⚕ Run AI Prediction"
        )


        predict_btn.clicked.connect(
            self.predict
        )


        layout.addWidget(
            predict_btn
        )



        biomarker_btn = QPushButton(
            "🧬 View Biomarkers"
        )


        biomarker_btn.clicked.connect(
            self.open_biomarkers
        )


        layout.addWidget(
            biomarker_btn
        )



        self.card = QFrame()


        card_layout = QVBoxLayout()



        self.result = QLabel(
            "Waiting for prediction..."
        )


        self.result.setStyleSheet(
            """
            font-size:24px;
            font-weight:bold;
            """
        )


        card_layout.addWidget(
            self.result
        )



        self.percent = QLabel(
            "0 %"
        )


        self.percent.setStyleSheet(
            """
            font-size:40px;
            font-weight:bold;
            """
        )


        card_layout.addWidget(
            self.percent
        )



        self.risk = QLabel(
            "Risk Level: Unknown"
        )


        card_layout.addWidget(
            self.risk
        )



        self.bar = QProgressBar()


        self.bar.setRange(
            0,
            100
        )


        self.bar.setValue(
            0
        )


        self.bar.setFixedHeight(
            35
        )


        card_layout.addWidget(
            self.bar
        )


        self.card.setLayout(
            card_layout
        )


        layout.addWidget(
            self.card
        )



        info = QLabel(
            """
AI Model:
Logistic Regression

Signature:
14 Biomarker Genes
"""
        )


        layout.addWidget(
            info
        )


        self.setLayout(
            layout
        )



    def open_file(self):

        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Patient CSV",
            "",
            "CSV Files (*.csv)"
        )


        if file:

            self.selected_file = file


            self.file_label.setText(
                "Selected:\n" + file
            )



    def prepare_patient(self):

        data = pd.read_csv(
            self.selected_file
        )


        if (
            "Gene" in data.columns
            and
            "Expression" in data.columns
        ):

            data = data.set_index(
                "Gene"
            ).T



        return data[
            self.genes
        ]



    def predict(self):

        try:

            patient = self.prepare_patient()


            x = self.scaler.transform(
                patient
            )


            prediction = self.model.predict(
                x
            )


            probability = (
                self.model.predict_proba(x)[0][1]
                *
                100
            )



            if prediction[0] == 1:

                status = "Parkinson Disease"

            else:

                status = "Healthy Control"



            self.result.setText(
                "Prediction:\n" + status
            )


            self.percent.setText(
                f"{probability:.2f}%"
            )


            self.bar.setValue(
                int(probability)
            )



            if probability >= 70:

                self.risk.setText(
                    "🔴 Risk Level: High"
                )

            else:

                self.risk.setText(
                    "🟢 Risk Level: Low"
                )



        except Exception as e:

            QMessageBox.critical(
                self,
                "Prediction Error",
                str(e)
            )



    def open_biomarkers(self):

        self.bio_window = BiomarkerPanel()

        self.bio_window.show()



if __name__ == "__main__":


    app = QApplication(
        sys.argv
    )


    window = MainWindow()

    window.show()


    sys.exit(
        app.exec()
    )