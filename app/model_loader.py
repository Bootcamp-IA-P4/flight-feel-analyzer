# app/model_loader.py
import joblib
import os

# --- Cargar NOMBRES de archivo desde variables de entorno ---
# Es importante que estas variables estén definidas en tu archivo .env
# Ejemplo en .env:
# MODEL_FILENAME=gradient_boosting_satisfaction_model.pkl
# SCALER_FILENAME=standard_scaler.pkl
# etc.
model_filename = os.environ.get('MODEL_FILENAME')
scaler_filename = os.environ.get('SCALER_FILENAME')
feature_order_filename = os.environ.get('FEATURE_ORDER_FILENAME')
cols_to_scale_filename = os.environ.get('COLS_TO_SCALE_FILENAME')

# --- Inicializar componentes ---
model = None
scaler = None
feature_order = None
cols_to_scale = None

# --- Determinar ruta base y directorio de modelos ---
# base_dir apuntará a C:\...\flight-feel-analyzer\app
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construye la ruta a la subcarpeta 'ml_models' DENTRO de 'app'
model_dir = os.path.join(base_dir, 'ml_models')

# --- Cargar Modelo ---
if model_filename:
    model_path = os.path.join(model_dir, model_filename)
    try:
        print(f"Intentando cargar modelo desde: {model_path} (definido por MODEL_FILENAME)")
        model = joblib.load(model_path)
        print("Modelo cargado.")
    except FileNotFoundError:
        print(f"Error crítico: Archivo de modelo '{model_filename}' no encontrado en '{model_dir}'. Verifique la variable MODEL_FILENAME en .env y la ubicación física del archivo.")
    except Exception as e:
        print(f"Error inesperado al cargar modelo '{model_filename}': {e}")
else:
    print("Error Crítico: La variable de entorno MODEL_FILENAME no está definida en .env. No se puede cargar el modelo.")

# --- Cargar Scaler ---
if scaler_filename:
    scaler_path = os.path.join(model_dir, scaler_filename)
    try:
        print(f"Intentando cargar scaler desde: {scaler_path} (definido por SCALER_FILENAME)")
        scaler = joblib.load(scaler_path)
        print("Scaler cargado.")
    except FileNotFoundError:
        print(f"Error crítico: Archivo de scaler '{scaler_filename}' no encontrado en '{model_dir}'. Verifique la variable SCALER_FILENAME en .env y la ubicación física.")
    except Exception as e:
        print(f"Error inesperado al cargar scaler '{scaler_filename}': {e}")
else:
    print("Error Crítico: La variable de entorno SCALER_FILENAME no está definida en .env. No se puede cargar el scaler.")

# --- Cargar Orden de Características ---
if feature_order_filename:
    feature_order_path = os.path.join(model_dir, feature_order_filename)
    try:
        print(f"Intentando cargar orden de características desde: {feature_order_path} (definido por FEATURE_ORDER_FILENAME)")
        feature_order = joblib.load(feature_order_path)
        print("Orden de características cargado.")
    except FileNotFoundError:
         print(f"Error crítico: Archivo de orden de características '{feature_order_filename}' no encontrado en '{model_dir}'. Verifique FEATURE_ORDER_FILENAME en .env y la ubicación física.")
    except Exception as e:
        print(f"Error inesperado al cargar orden de características '{feature_order_filename}': {e}")
else:
    print("Error Crítico: La variable de entorno FEATURE_ORDER_FILENAME no está definida en .env.")

# --- Cargar Columnas a Escalar ---
if cols_to_scale_filename:
    cols_to_scale_path = os.path.join(model_dir, cols_to_scale_filename)
    try:
        print(f"Intentando cargar columnas a escalar desde: {cols_to_scale_path} (definido por COLS_TO_SCALE_FILENAME)")
        cols_to_scale = joblib.load(cols_to_scale_path)
        print("Columnas a escalar cargadas.")
    except FileNotFoundError:
        print(f"Error crítico: Archivo de columnas a escalar '{cols_to_scale_filename}' no encontrado en '{model_dir}'. Verifique COLS_TO_SCALE_FILENAME en .env y la ubicación física.")
    except Exception as e:
         print(f"Error inesperado al cargar columnas a escalar '{cols_to_scale_filename}': {e}")
else:
    print("Error Crítico: La variable de entorno COLS_TO_SCALE_FILENAME no está definida en .env.")

# --- Función para acceder a los componentes (sin cambios) ---
def get_model_components():
    """Retorna los componentes cargados del modelo."""
    # Esta función ahora depende de que las variables globales (model, scaler, etc.)
    # se hayan cargado correctamente arriba. Si alguna falló, será None.
    if not all([model, scaler, feature_order, cols_to_scale]):
        print("Advertencia: Uno o más componentes del modelo (model, scaler, feature_order, cols_to_scale) no se cargaron correctamente debido a errores previos (revisar logs).")
    return model, scaler, feature_order, cols_to_scale