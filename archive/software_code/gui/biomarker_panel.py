from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QListWidget
)

import os


class BiomarkerPanel(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Parkinson AI Biomarkers"
        )

        self.resize(
            500,
            500
        )

        self.load_genes()

        self.build_ui()



    def load_genes(self):

        base_path = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )


        genes_path = os.path.join(
            base_path,
            "models",
            "final_genes.txt"
        )


        with open(
            genes_path,
            "r"
        ) as f:

            self.genes = [
                g.strip()
                for g in f.readlines()
            ]



    def build_ui(self):

        layout = QVBoxLayout()


        title = QLabel(
            "🧬 Important Parkinson Biomarkers\n"
            "AI Selected Signature Genes"
        )


        title.setStyleSheet(
            """
            font-size:22px;
            font-weight:bold;
            """
        )


        layout.addWidget(title)



        self.list_widget = QListWidget()


        for gene in self.genes:

            self.list_widget.addItem(
                "🔹 " + gene
            )


        layout.addWidget(
            self.list_widget
        )


        info = QLabel(
            """
Model:
Logistic Regression

Selected genes:
14 Biomarkers
"""
        )


        layout.addWidget(
            info
        )


        self.setLayout(
            layout
        )