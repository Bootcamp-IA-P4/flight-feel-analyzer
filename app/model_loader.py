# app/model_loader.py
import joblib
import os

# Obtener el directorio del script actual (model_loader.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models')

# Usar variables de entorno si están definidas, si no, usar valores por defecto
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'model.pkl') # <- NOMBRE USADO EN TU SCRIPT model_satisfaction.py
FEATURE_ORDER_FILENAME = os.getenv('FEATURE_ORDER_FILENAME', 'feature_order.joblib')

MODEL_FILE = os.path.join(MODEL_DIR, MODEL_FILENAME)
FEATURE_ORDER_FILE = os.path.join(MODEL_DIR, FEATURE_ORDER_FILENAME)

_model = None
_feature_order = None
_components_loaded = False

def _load_components():
    global _model, _feature_order, _components_loaded
    print(f"Intentando cargar modelo desde: {MODEL_FILE} (definido por MODEL_FILENAME)")
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Error crítico: Archivo de modelo '{MODEL_FILENAME}' no encontrado en '{MODEL_DIR}'. Verifique la variable MODEL_FILENAME en .env y la ubicación física del archivo.")
            _model = None # Asegurar que es None si no se encuentra
        else:
            _model = joblib.load(MODEL_FILE)
            print("Modelo cargado.")

        print(f"Intentando cargar orden de características desde: {FEATURE_ORDER_FILE} (definido por FEATURE_ORDER_FILENAME)")
        if not os.path.exists(FEATURE_ORDER_FILE):
            print(f"Error crítico: Archivo de orden de características '{FEATURE_ORDER_FILENAME}' no encontrado en '{MODEL_DIR}'. Verifique FEATURE_ORDER_FILENAME en .env y la ubicación física.")
            _feature_order = None # Asegurar que es None si no se encuentra
        else:
            _feature_order = joblib.load(FEATURE_ORDER_FILE)
            print("Orden de características cargado.")
            
        if _model and _feature_order:
            _components_loaded = True
            print("Componentes (modelo y orden de características) cargados exitosamente.")
        else:
            print("Uno o más componentes (modelo u orden de características) fallaron al cargar.")
            _components_loaded = False # Mantener False si algo falla
            # Podrías lanzar una excepción aquí si es un fallo crítico que impida iniciar la app
            # raise RuntimeError("Fallo crítico al cargar componentes del modelo ML.")

    except Exception as e:
        print(f"Error excepcional durante la carga de componentes: {e}")
        _model, _feature_order = None, None # Reiniciar en caso de error
        _components_loaded = False

# ESTA ES LA FUNCIÓN QUE routes.py INTENTA IMPORTAR
def get_model_and_feature_order():
    global _components_loaded
    if not _components_loaded: # Cargar solo si no se ha intentado o si falló antes
        _load_components()
    return _model, _feature_order

# Opcional: realizar una carga inicial cuando el módulo se importa por primera vez
# if not _components_loaded:
# _load_components()