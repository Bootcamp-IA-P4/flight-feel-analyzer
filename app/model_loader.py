# app/model_loader.py
import joblib
import os

# Obtener el directorio del script actual (model_loader.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'model.pkl') # Nombre del modelo de tu compañera
MODEL_FILE = os.path.join(MODEL_DIR, MODEL_FILENAME)

_model = None
_model_loaded_successfully = False # Bandera para saber si la carga fue exitosa

def _load_model():
    global _model, _model_loaded_successfully
    print(f"Intentando cargar modelo (Pipeline completo) desde: {MODEL_FILE}")
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Error crítico: Archivo de modelo '{MODEL_FILENAME}' no encontrado en '{MODEL_DIR}'.")
            _model = None
            _model_loaded_successfully = False
        else:
            _model = joblib.load(MODEL_FILE)
            print("Modelo (Pipeline completo) cargado exitosamente.")
            _model_loaded_successfully = True

    except Exception as e:
        print(f"Error excepcional durante la carga del modelo: {e}")
        _model = None
        _model_loaded_successfully = False

# Función para obtener el modelo
def get_model():
    global _model_loaded_successfully
    if not _model_loaded_successfully: # Solo intentar cargar si no se ha cargado o falló antes
        _load_model()
    return _model

# Opcional: Carga inicial al importar el módulo.
# _load_model() # Descomenta si quieres que se cargue al inicio de la appS