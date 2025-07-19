
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import sys
import os

# Ensure diabetes.csv exists in the script directory
if not os.path.exists("diabetes.csv"):
    app = QApplication([])
    QMessageBox.critical(None, "File Missing", "Make sure 'diabetes.csv' is in the same folder as this script.")
    sys.exit()

# Load and train model
df = pd.read_csv("diabetes.csv")

# Validate features
required_features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
if not all(feature in df.columns for feature in required_features):
    app = QApplication([])
    QMessageBox.critical(None, "CSV Error", "The CSV file does not contain the required columns.")
    sys.exit()

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# PyQt5 GUI application
class DiabetesPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diabetes Risk Predictor")
        self.setGeometry(100, 100, 420, 640)
        self.setStyleSheet("background-color: #1e1e2f; border-radius: 10px;")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        title = QLabel("Diabetes Risk Predictor")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.inputs = []
        self.labels = [
            ("Pregnancies", "Pregnancies"),
            ("Glucose", "Glucose"),
            ("Blood Pressure", "BloodPressure"),
            ("Skin Thickness", "SkinThickness"),
            ("Insulin", "Insulin"),
            ("BMI", "BMI"),
            ("Diabetes Pedigree", "DiabetesPedigreeFunction"),
            ("Age", "Age")
        ]
        self.feature_order = [key for label, key in self.labels]

        for label_text, _ in self.labels:
            hbox = QHBoxLayout()
            label = QLabel(label_text + ":")
            label.setFont(QFont("Arial", 12))
            label.setFixedWidth(150)
            label.setStyleSheet("color: #DDDDDD;")
            line_edit = QLineEdit()
            line_edit.setStyleSheet(
                "padding: 8px; border: 1px solid #444; border-radius: 6px; background-color: #2c2c3c; color: white;"
            )
            hbox.addWidget(label)
            hbox.addWidget(line_edit)
            self.inputs.append(line_edit)
            layout.addLayout(hbox)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setStyleSheet("color: white;")
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        predict_button = QPushButton("Check Diabetes Risk")
        predict_button.setStyleSheet("""
            QPushButton {
                background-color: #007BFF; 
                color: white; 
                padding: 12px; 
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
        """)
        predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(predict_button)

        self.setLayout(layout)

    def make_prediction(self):
        try:
            data = [float(field.text()) for field in self.inputs]
            input_array = np.array([data])
            prediction = model.predict(input_array)[0]
            if prediction == 1:
                self.result_label.setText("High Risk of Diabetes")
                self.result_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.result_label.setText("Low Risk of Diabetes")
                self.result_label.setStyleSheet("color: green; font-weight: bold;")
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numeric values.")

# Run the application
app = QApplication(sys.argv)
window = DiabetesPredictor()
window.show()
app.exec_()
