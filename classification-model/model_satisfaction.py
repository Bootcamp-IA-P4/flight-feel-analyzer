import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV # train_test_split importado
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- CONSTRUCCIÓN DE RUTA CORRECTA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
csv_path_processed = os.path.join(data_dir, "processed_data.csv")
# ------------------------------------

try:
    df_model = pd.read_csv(csv_path_processed)
    print(f"Datos cargados exitosamente desde: {csv_path_processed}")
except FileNotFoundError:
    print(f"Error: '{csv_path_processed}' no encontrado. Verifica la ruta y la estructura del proyecto.")
    exit()

print("Columnas originales de processed_data.csv:", df_model.columns.tolist())

if 'Unnamed: 0' in df_model.columns:
    print("Eliminando 'Unnamed: 0' de df_model.")
    df_model = df_model.drop(columns=['Unnamed: 0'])
if 'Unnamed: 0.1' in df_model.columns:
    print("Eliminando 'Unnamed: 0.1' de df_model.")
    df_model = df_model.drop(columns=['Unnamed: 0.1'])

if 'satisfaction' not in df_model.columns:
    print("Error: La columna 'satisfaction' no se encuentra en el dataset.")
    exit()

X = df_model.drop(["satisfaction"], axis=1)
y = df_model["satisfaction"]

print("Columnas usadas para X (después de eliminar Unnamed: 0 y target):", X.columns.tolist())

app_ml_models_dir = os.path.join(project_root, "app", "ml_models")
if not os.path.exists(app_ml_models_dir):
    os.makedirs(app_ml_models_dir)
    print(f"Directorio creado: {app_ml_models_dir}")

feature_order_path = os.path.join(app_ml_models_dir, "feature_order.joblib")
joblib.dump(list(X.columns), feature_order_path)
print(f"Orden de características guardado en: {feature_order_path}")

model_app_path = os.path.join(app_ml_models_dir, "model.pkl")

# *** MOVER train_test_split AQUÍ, ANTES de usar X_train, y_train ***
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Datos divididos: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
# ********************************************************************

param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=10,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Iniciando RandomizedSearchCV...")
random_search.fit(X_train, y_train) # Ahora X_train y y_train están definidos
print("Mejores hiperparámetros:", random_search.best_params_)
best_rf_model = random_search.best_estimator_

joblib.dump(best_rf_model, model_app_path, compress=3)
print(f"Modelo guardado en: {model_app_path}")

# Predicciones
y_pred_test = best_rf_model.predict(X_test)
y_pred_train = best_rf_model.predict(X_train)
y_proba_test = best_rf_model.predict_proba(X_test)[:, 1]
y_proba_train = best_rf_model.predict_proba(X_train)[:, 1]

# === Detección de Overfitting ===
f1_train = f1_score(y_train, y_pred_train, pos_label=1, zero_division=0)
f1_test = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
overfitting = False
overfitting_threshold = 0.05
if f1_test < (f1_train - overfitting_threshold):
    overfitting = True

overfitting_percentage = 0.0
if f1_train > 0:
    overfitting_percentage = 100 * (f1_train - f1_test) / f1_train
else:
    print("F1 score en entrenamiento es 0, no se puede calcular overfitting relativo.")

# === Métricas en entrenamiento ===
print("\n=== Métricas del modelo (Entrenamiento) ===")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision: {precision_score(y_train, y_pred_train, pos_label=1, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_train, y_pred_train, pos_label=1, zero_division=0):.4f}")
print(f"F1 Score: {f1_train:.4f}")
print(f"ROC AUC: {roc_auc_score(y_train, y_proba_train):.4f}")
if f1_train > 0:
    print(f"Overfitting: {overfitting_percentage:.2f}%")
    print(f"Overfitting detected based on F1: {overfitting}")
else:
    print("Overfitting F1 no aplicable.")

# === Métricas en prueba ===
print("\n=== Métricas del modelo (Test) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test, pos_label=1, zero_division=0):.4f}")
print(f"F1 Score: {f1_test:.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
if f1_train > 0:
    print(f"Overfitting: {overfitting_percentage:.2f}%")
    print(f"Overfitting detected based on F1: {overfitting}")
else:
    print("Overfitting F1 no aplicable.")