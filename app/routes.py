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
# El primer argumento 'main' es el nombre del blueprint.
# El segundo __name__ ayuda a Flask a encontrar recursos como templates.
main_bp = Blueprint('main', __name__)

# Cargar componentes del modelo al iniciar (esto se hace cuando se importa model_loader)
model, scaler, feature_order, cols_to_scale = get_model_components()

# --- Definir rutas usando el Blueprint (@main_bp.route) ---
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
    if not all([model, scaler, feature_order, cols_to_scale]):
        flash('Error: El modelo o sus componentes no están cargados correctamente. Contacte al administrador.', 'danger')
        # Usa el nombre del endpoint del blueprint: 'main.index'
        return redirect(url_for('main.index'))

    form_data = {}
    try:
        # --- 1. Recopilar datos del formulario ---
        form_data = request.form.to_dict()
        print("Datos recibidos del formulario:", form_data)

        personal_data = {
            'customer_name': form_data.get('customer_name'),
            'customer_email': form_data.get('customer_email'),
            'customer_dni': form_data.get('customer_dni'),
            'customer_phone': form_data.get('customer_phone')
        }

        # --- 2. Preprocesar los datos PARA EL MODELO ML ---
        model_input_data = {k: v for k, v in form_data.items() if k not in personal_data}
        input_df = pd.DataFrame([model_input_data])

        model_numeric_cols_original = [col for col in cols_to_scale if col in input_df.columns and col != 'Total Delay in Minutes']

        for col in model_numeric_cols_original:
             input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        if input_df[model_numeric_cols_original].isnull().any().any():
             flash('Error en la entrada: Asegúrate de que todos los campos numéricos de valoración/vuelo contengan números válidos.', 'danger')
             return redirect(url_for('main.index'))

        input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
        input_df['Customer Type'] = input_df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0}).astype(int)
        input_df['Type of Travel'] = input_df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1}).astype(int)
        input_df['Class'] = input_df['flight_class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2}).astype(int)

        input_df['Total Delay in Minutes'] = input_df['Departure Delay in Minutes'] + input_df['Arrival Delay in Minutes']

        try:
            input_df_ordered = input_df[feature_order]
        except KeyError as ke:
            missing_cols = set(feature_order) - set(input_df.columns)
            print(f"Error: Faltan columnas esperadas por el modelo: {missing_cols}")
            flash(f"Error interno: Faltan datos esperados para el modelo: {missing_cols}", 'danger')
            return redirect(url_for('main.index'))

        input_df_ordered.loc[:, cols_to_scale] = scaler.transform(input_df_ordered[cols_to_scale])

        print("DataFrame preprocesado listo para predicción:\n", input_df_ordered)

        if input_df_ordered.isnull().any().any():
             flash('Error interno después del preprocesamiento. Contacta al administrador.', 'danger')
             print("Error: NaNs encontrados ANTES de la predicción", input_df_ordered[input_df_ordered.isnull().any(axis=1)])
             return redirect(url_for('main.index'))

        # --- 3. Realizar Predicción ---
        prediction = model.predict(input_df_ordered)[0]
        prediction_proba = model.predict_proba(input_df_ordered)[0]
        probability_satisfied = prediction_proba[1]
        satisfaction_result = "Satisfecho ✅" if prediction == 1 else "Neutral o Insatisfecho ❌"

        print(f"Predicción: {prediction} ({satisfaction_result}), Probabilidad Satisfecho: {probability_satisfied:.4f}")

        # --- 4. Guardar en Base de Datos ---
        new_entry = NewFlightData(
            customer_name=personal_data['customer_name'],
            customer_email=personal_data['customer_email'],
            customer_dni=personal_data['customer_dni'],
            customer_phone=personal_data['customer_phone'],
            gender=form_data.get('Gender'),
            customer_type=form_data.get('Customer Type'),
            age=int(form_data.get('Age', 0)),
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
        flash(f'Error en los datos ingresados ({ve}). Asegúrate de que los números sean válidos.', 'danger')
        return redirect(url_for('main.index')) # Referencia al endpoint del blueprint
    except KeyError as ke:
         print(f"Error de Clave: Falta la clave {ke}.")
         print(traceback.format_exc())
         flash(f'Error interno: Falta información requerida ({ke}).', 'danger')
         return redirect(url_for('main.index')) # Referencia al endpoint del blueprint
    except Exception as e:
        print(f"Error inesperado durante la predicción o guardado: {e}")
        print(traceback.format_exc())
        try:
            db.session.rollback()
            print("Rollback de la sesión de base de datos realizado.")
        except Exception as rb_e:
            print(f"Error adicional durante el rollback: {rb_e}")
        flash(f'Ocurrió un error inesperado al procesar tu solicitud.', 'danger')
        return redirect(url_for('main.index')) # Referencia al endpoint del blueprint