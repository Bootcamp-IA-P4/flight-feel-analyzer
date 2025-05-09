import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

current_dir = os.getcwd()  # Obtener el directorio actual
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data")
file_path = os.path.join(data_dir, "")
csv_path_processed = os.path.join(data_dir, "processed_data.csv")
df_model = pd.read_csv(csv_path_processed)

X = df_model.drop(["satisfaction"], axis=1) 
y = df_model["satisfaction"]  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

best_rf = RandomForestClassifier(random_state=42)

param_distributions = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [4, 6, 8],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=best_rf,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='roc_auc',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
print("Mejores hiperparámetros:", random_search.best_params_)
best_rf = random_search.best_estimator_  # best_estimator_ es un atributo de RandomizedSearchCV que contiene el mejor modelo encontrado durante la búsqueda de hiperparámetros, aquí le decimos Guarda el mejor modelo encontrado por RandomizedSearchCV en la variable best_rf.

# Entrenar el modelo con los mejores hiperparámetros
best_rf.fit(X_train, y_train)

# Hacer predicciones
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

# Guardar el modelo
model_path = "model.pkl"
joblib.dump(best_rf, model_path, compress=3)
print(f"Modelo guardado en: {model_path}")

# Predicciones
y_pred_test = best_rf.predict(X_test)
y_pred_train = best_rf.predict(X_train)
y_proba_test = best_rf.predict_proba(X_test)[:, 1]
y_proba_train = best_rf.predict_proba(X_train)[:, 1]

# === Detección de Overfitting ===
f1_train = f1_score(y_train, y_pred_train, pos_label=1)
f1_test = f1_score(y_test, y_pred_test, pos_label=1)
overfitting = False
overfitting_threshold = 0.05  
if f1_test < (f1_train - overfitting_threshold):
    overfitting = True
overfitting_percentage = 100 * (f1_train - f1_test) / f1_train

# === Métricas en entrenamiento ===
print("\n=== Métricas del modelo (Entrenamiento) ===")
print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Precision: {precision_score(y_train, y_pred_train, pos_label=1):.4f}")
print(f"Recall: {recall_score(y_train, y_pred_train, pos_label=1):.4f}")
print(f"F1 Score: {f1_score(y_train, y_pred_train, pos_label=1):.4f}")
print(f"ROC AUC: {roc_auc_score(y_train, y_proba_train):.4f}")
print(f"Overfitting: {overfitting_percentage:.2f}%")
print(f"Overfitting detected: {overfitting_percentage:.2f}%" if overfitting else "No overfitting detected")


# === Métricas en prueba ===
print("\n=== Métricas del modelo (Test) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_test, pos_label=1):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_test, pos_label=1):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_test, pos_label=1):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_test):.4f}")
print(f"Overfitting: {overfitting_percentage:.2f}%")
print(f"Overfitting detected: {overfitting_percentage:.2f}%" if overfitting else "No overfitting detected")

