import unittest
import pandas as pd
import joblib
import sys
import os

base_path = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo
model_path = os.path.join(base_path, "..", "app", "ml_models", "model.pkl")
# Ruta al dataset
data_path = os.path.join(base_path, "..", "classification-model", "data", "processed_data.csv")

class TestModelSatisfaction(unittest.TestCase):
    def setUp(self):
        # Cargamos el modelo entrenado
        self.model = joblib.load(model_path)

        # Cargar los datos de prueba
        data_path = os.path.join(base_path, "..", "data", "processed_data.csv")
        self.df = pd.read_csv(data_path)
        self.X = self.df.drop("satisfaction", axis=1)
        self.y = self.df["satisfaction"]

    def test_model_loaded(self):
        self.assertIsNotNone(self.model, "No se cargó el modelo")

    def test_prediction_shape(self):
        y_pred = self.model.predict(self.X)
        self.assertEqual(len(y_pred), len(self.X), "El número de predicciones no coincide")

    def test_prediction_proba_range(self):
        y_proba = self.model.predict_proba(self.X)[:, 1]
        self.assertTrue(all(0 <= p <= 1 for p in y_proba),"Las probabilidades de predicción deben estar entre 0 y 1")

    def test_model_has_predict_method(self):
        self.assertTrue(hasattr(self.model, "predict"), "El modelo no tiene el método predict")

    def test_data_not_empty(self):
        self.assertFalse(self.df.empty, "El DataFrame de datos está vacío")

    def test_input_columns(self):
        expected_columns = ["Gender", "Customer Type", "Age", "Type of Travel", "Class",
        "Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient",
        "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
        "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
        "Departure Delay in Minutes"]
        self.assertListEqual(list(self.X.columns), expected_columns, "Las columnas de entrada no coinciden con las esperadas")

if __name__ == '__main__':
    unittest.main()