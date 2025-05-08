# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import traceback

# Importar componentes necesarios del PAQUETE ACTUAL (app)
from . import db # Accede a la instancia db creada en __init__.py
from .model_loader import get_model_components
from .models.flight_data_model import NewFlightData

# --- Crear un Blueprint ---
main_bp = Blueprint('main', __name__)

# Cargar componentes del modelo al iniciar
model, scaler, feature_order, cols_to_scale = get_model_components()

# --- Definir rutas usando el Blueprint ---
@main_bp.route('/')
def index():
    """Muestra el formulario principal."""
    return render_template('index.html')

@main_bp.route('/about')
def about():
    """Muestra la página 'Acerca del equipo'."""
    return render_template('about-team.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    """Recibe datos del formulario, predice y guarda en DB."""
    if not all([model, scaler is not None, feature_order, cols_to_scale is not None]): # scaler y cols_to_scale pueden ser None/lista vacía
        if model is None:
            flash('Error: El componente del modelo ML no está cargado. Contacte al administrador.', 'danger')
        if scaler is None and cols_to_scale: # Si se esperaban columnas para escalar pero el scaler es None
            flash('Error: El componente Scaler del modelo no está cargado pero se esperaban columnas a escalar. Contacte al administrador.', 'danger')
        if not feature_order:
            flash('Error: El orden de características para el modelo no está cargado. Contacte al administrador.', 'danger')
        # No es un error si cols_to_scale es una lista vacía y scaler es None, significa que no hay nada que escalar.
        return redirect(url_for('main.index'))

    form_data = {}
    try:
        # --- 1. Recopilar datos del formulario ---
        form_data = request.form.to_dict()
        print("\n--- Datos Recibidos del Formulario ---")
        print(form_data)

        personal_data = {
            'customer_name': form_data.get('customer_name'),
            'customer_email': form_data.get('customer_email'),
            'customer_dni': form_data.get('customer_dni'),
            'customer_phone': form_data.get('customer_phone')
        }

        # --- 2. Preprocesar los datos PARA EL MODELO ML ---
        print("\n--- Inicio Preprocesamiento ---")
        model_input_data = {k: v for k, v in form_data.items() if k not in personal_data}
        input_df = pd.DataFrame([model_input_data])
        print("DataFrame inicial desde form (sin datos personales):\n", input_df.head().to_string())

        # Columnas que SON NUMÉRICAS directamente desde el formulario y deben serlo para el modelo
        direct_numeric_cols_from_form = [
            'Age', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service',
            'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'
        ]
        
        cols_to_convert_numeric = [col for col in direct_numeric_cols_from_form if col in input_df.columns]
        print("Columnas a convertir a numérico directamente desde el form:", cols_to_convert_numeric)

        for col in cols_to_convert_numeric:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        print("DataFrame después de to_numeric (solo directas):\n", input_df.head().to_string())

        if input_df[cols_to_convert_numeric].isnull().any().any():
            nan_cols = input_df[cols_to_convert_numeric].isnull().sum()
            error_message_nan = "Error: Los siguientes campos deben ser números válidos: " + ", ".join(nan_cols[nan_cols > 0].index.tolist())
            print(error_message_nan)
            flash(error_message_nan, 'danger')
            return redirect(url_for('main.index'))

        # Mapear categóricas
        print("Mapeando categóricas...")
        expected_categorical_keys_from_form = {'Gender': 'Gender',
                                               'Customer Type': 'Customer Type',
                                               'Type of Travel': 'Type of Travel',
                                               'flight_class': 'Class'} # form_key: model_col_name
        
        for form_key, model_col_name in expected_categorical_keys_from_form.items():
            if form_key not in input_df.columns:
                print(f"Error: Falta la clave '{form_key}' del formulario en input_df.")
                flash(f"Error: Falta el campo para '{model_col_name}'. Por favor, completa todos los campos requeridos.", 'danger')
                return redirect(url_for('main.index'))

        input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
        input_df['Customer Type'] = input_df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
        input_df['Type of Travel'] = input_df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        input_df['Class'] = input_df['flight_class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})

        cols_after_mapping = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col_am in cols_after_mapping:
            if input_df[col_am].isnull().any():
                original_form_key = next((fk for fk, mcn in expected_categorical_keys_from_form.items() if mcn == col_am), col_am)
                print(f"Error: NaN encontrado en '{col_am}' (desde form key '{original_form_key}') después del mapeo. Valor no esperado del formulario.")
                flash(f"Error: Valor no válido para el campo '{original_form_key}'.", "danger")
                return redirect(url_for('main.index'))
            input_df[col_am] = input_df[col_am].astype(int)

        print("DataFrame después de mapeo categórico y conversión a int:\n", input_df.head().to_string())

        # Calcular característica derivada 'Total Delay in Minutes'
        # Asegurar que las columnas base para el cálculo existan y sean numéricas
        delay_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
        for dc in delay_cols:
            if dc not in input_df.columns: # Ya deberían ser numéricas por el paso anterior
                 print(f"Error: Falta la columna de retraso '{dc}' en input_df.")
                 flash(f"Error: Falta el campo '{dc}'. Por favor, completa todos los campos.", 'danger')
                 return redirect(url_for('main.index'))
            # Ya son numéricas
        
        input_df['Total Delay in Minutes'] = input_df['Departure Delay in Minutes'] + input_df['Arrival Delay in Minutes']
        print("DataFrame con Total Delay:\n", input_df.head().to_string())


        # Preparar el DataFrame final para el modelo usando feature_order
        print("Preparando DataFrame final para el modelo según feature_order...")
        input_df_final_model = pd.DataFrame(index=[0], columns=feature_order)

        for col_fo in feature_order:
            if col_fo in input_df.columns:
                input_df_final_model[col_fo] = input_df[col_fo].values
            else:
                # Esto no debería pasar si feature_order es correcto y el preproc de aquí lo genera todo
                print(f"ADVERTENCIA CRÍTICA: La columna '{col_fo}' esperada por el modelo NO se encuentra en input_df después del preprocesamiento.")
                # Decidir cómo manejar: Error o imputación muy básica. Imputar puede dar malas predicciones.
                # Es mejor lanzar error si una característica fundamental falta.
                flash(f"Error interno de preprocesamiento: Falta la característica '{col_fo}'. Contacte al administrador.", "danger")
                return redirect(url_for('main.index'))
                # input_df_final_model[col_fo] = 0 # Imputación de último recurso NO RECOMENDADA

        print("DataFrame final (input_df_final_model) ANTES de escalado:\n", input_df_final_model.head().to_string())
        print("Tipos de datos ANTES de escalado:\n", input_df_final_model.dtypes)


        # Aplicar el Escalado
        # cols_to_scale es la lista de columnas que el scaler ESPERA (con la que fue ajustado)
        if scaler is not None and cols_to_scale:
            print(f"\nEscalando estas {len(cols_to_scale)} columnas (definidas en cols_to_scale.joblib): {cols_to_scale}")
            
            # Verificar que todas las columnas a escalar existan en input_df_final_model
            missing_cols_for_scaling = [col for col in cols_to_scale if col not in input_df_final_model.columns]
            if missing_cols_for_scaling:
                print(f"ERROR: Faltan columnas necesarias para el escalado en el DataFrame: {missing_cols_for_scaling}")
                flash("Error interno: Faltan columnas para el escalado. Contacte al administrador.", 'danger')
                return redirect(url_for('main.index'))

            # Seleccionar solo las columnas que se van a escalar y asegurar que sean float
            df_subset_to_scale = input_df_final_model[cols_to_scale].astype(float)

            if df_subset_to_scale.isnull().values.any():
                nan_in_scaling_subset = df_subset_to_scale.isnull().sum()
                print("ERROR: NaNs detectados en el subconjunto de columnas A ESCALAR, ANTES de scaler.transform:")
                print(nan_in_scaling_subset[nan_in_scaling_subset > 0])
                flash('Error interno de preprocesamiento (NaNs en datos a escalar). Contacte al administrador.', 'danger')
                return redirect(url_for('main.index'))
            
            try:
                scaled_values = scaler.transform(df_subset_to_scale)
                scaled_df_temp = pd.DataFrame(scaled_values, index=input_df_final_model.index, columns=cols_to_scale)
                input_df_final_model.update(scaled_df_temp) # Actualiza solo las columnas escaladas en el df original
                print("Escalado aplicado.")
            except ValueError as ve:
                print(f"Error durante scaler.transform: {ve}")
                print(traceback.format_exc())
                flash('Error interno durante el preprocesamiento (escalado). Verifique la consistencia de las características.', 'danger')
                return redirect(url_for('main.index'))
            except Exception as scaling_error:
                print(f"Error inesperado durante el escalado: {scaling_error}")
                print(traceback.format_exc())
                flash('Error interno general durante el preprocesamiento (escalado).', 'danger')
                return redirect(url_for('main.index'))
        elif scaler is None and cols_to_scale:
             print("Advertencia: Se definieron columnas para escalar, pero el objeto Scaler no está cargado. Se omitirá el escalado.")
        else: # scaler es None y/o cols_to_scale está vacío/None
            print("No se realizará escalado (scaler no disponible o no hay columnas para escalar).")


        print("\nDataFrame FINAL listo para predicción (input_df_final_model):\n", input_df_final_model.head().to_string())
        print("Tipos finales:\n", input_df_final_model.dtypes)

        if input_df_final_model[feature_order].isnull().values.any(): # Verificar NaNs solo en las columnas que usará el modelo
            nan_final_check = input_df_final_model[feature_order].isnull().sum()
            print("ERROR CRÍTICO: NaNs detectados en el DataFrame final (columnas para el modelo) ANTES de la predicción.")
            print(nan_final_check[nan_final_check > 0])
            flash('Error interno de datos (NaNs) antes de la predicción. Contacte al administrador.', 'danger')
            return redirect(url_for('main.index'))


        # --- 3. Realizar Predicción ---
        print("\n--- Iniciando Predicción ---")
        # El modelo espera las columnas en el orden exacto de 'feature_order'
        prediction_input = input_df_final_model[feature_order]
        prediction = model.predict(prediction_input)[0]
        prediction_proba = model.predict_proba(prediction_input)[0]
        
        probability_satisfied = prediction_proba[1] # Asumiendo que la clase 1 es 'satisfied'
        satisfaction_result = "Satisfecho ✅" if prediction == 1 else "Neutral o Insatisfecho ❌"

        print(f"Predicción: {prediction} ({satisfaction_result}), Probabilidad Satisfecho: {probability_satisfied:.4f}")

        # --- 4. Guardar en Base de Datos ---
        print("\n--- Guardando en Base de Datos ---")
        new_entry = NewFlightData(
            customer_name=personal_data.get('customer_name'),
            customer_email=personal_data.get('customer_email'),
            customer_dni=personal_data.get('customer_dni'),
            customer_phone=personal_data.get('customer_phone'),
            # Guardar los valores originales del formulario para la DB si se prefiere
            gender=form_data.get('Gender'),
            customer_type=form_data.get('Customer Type'),
            age=int(form_data.get('Age', 0)), # Asegurar int
            type_of_travel=form_data.get('Type of Travel'),
            flight_class=form_data.get('flight_class'),
            flight_distance=int(form_data.get('Flight Distance', 0)),
            inflight_wifi_service=int(form_data.get('Inflight wifi service', 0)),
            departure_arrival_time_convenient=int(form_data.get('Departure/Arrival time convenient', 0)),
            ease_of_online_booking=int(form_data.get('Ease of Online booking', 0)),
            gate_location=int(form_data.get('Gate location', 0)),
            food_and_drink=int(form_data.get('Food and drink', 0)),
            online_boarding=int(form_data.get('Online boarding', 0)),
            seat_comfort=int(form_data.get('Seat comfort', 0)),
            inflight_entertainment=int(form_data.get('Inflight entertainment', 0)),
            onboard_service=int(form_data.get('On-board service', 0)),
            leg_room_service=int(form_data.get('Leg room service', 0)),
            baggage_handling=int(form_data.get('Baggage handling', 0)),
            checkin_service=int(form_data.get('Checkin service', 0)),
            inflight_service=int(form_data.get('Inflight service', 0)),
            cleanliness=int(form_data.get('Cleanliness', 0)),
            departure_delay_in_minutes=int(form_data.get('Departure Delay in Minutes', 0)),
            arrival_delay_in_minutes=int(form_data.get('Arrival Delay in Minutes', 0)),
            predicted_satisfaction=int(prediction),
            prediction_probability_satisfied=float(probability_satisfied)
        )
        db.session.add(new_entry)
        db.session.commit()
        print(f"Datos guardados en la base de datos con ID: {new_entry.id}")
        flash('Predicción realizada y datos guardados exitosamente!', 'success')

        # --- 5. Mostrar Resultado ---
        return render_template('predict.html',
                               prediction_text=satisfaction_result,
                               probability=f"{probability_satisfied:.2%}")

    except ValueError as ve:
        print(f"Error de Valor/Tipo al procesar datos: {ve}")
        print(traceback.format_exc())
        flash(f'Error en los datos ingresados. Detalles: {ve}. Por favor, revisa los campos numéricos.', 'danger')
        return redirect(url_for('main.index'))
    except KeyError as ke:
         print(f"Error de Clave: Es posible que falte la clave '{ke}' del formulario o un problema en el mapeo.")
         print(traceback.format_exc())
         flash(f"Error interno: Falta información requerida ('{ke}') o error de configuración. Por favor, revisa todos los campos.", 'danger')
         return redirect(url_for('main.index'))
    except Exception as e:
        print(f"Error inesperado durante la predicción o guardado: {e}")
        print(traceback.format_exc())
        try:
            db.session.rollback() # Importante para deshacer cambios parciales en DB
            print("Rollback de la sesión de base de datos realizado.")
        except Exception as rb_e:
            print(f"Error adicional durante el rollback: {rb_e}")
        flash(f'Ocurrió un error inesperado ({type(e).__name__}) al procesar tu solicitud. Intenta de nuevo o contacta al administrador.', 'danger')
        return redirect(url_for('main.index'))