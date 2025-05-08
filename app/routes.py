# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import traceback

from . import db
from .model_loader import get_model_and_feature_order # Carga el modelo y el orden de características
from .models.flight_data_model import NewFlightData

main_bp = Blueprint('main', __name__)
model, feature_order = get_model_and_feature_order()

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about-team.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    if not model or not feature_order:
        flash('Error: El modelo o el orden de características no están cargados. Contacte al administrador.', 'danger')
        return redirect(url_for('main.index'))

    form_data = {}
    try:
        form_data = request.form.to_dict()
        print("\n--- Datos Recibidos del Formulario ---")
        print(form_data)

        personal_data = {
            'customer_name': form_data.get('customer_name'),
            'customer_email': form_data.get('customer_email'),
            'customer_dni': form_data.get('customer_dni'),
            'customer_phone': form_data.get('customer_phone')
        }
        
        model_input_data = {k: v for k, v in form_data.items() if k not in personal_data}
        input_df = pd.DataFrame([model_input_data])
        print("DataFrame inicial desde form (sin datos personales):\n", input_df.head().to_string())

        # Columnas que SON NUMÉRICAS directamente desde el formulario
        # Basado en el log, 'Arrival Delay in Minutes' SÍ es usada por el modelo.
        direct_numeric_cols_from_form = [
            'Age', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service',
            'Cleanliness', 'Departure Delay in Minutes',
            'Arrival Delay in Minutes' # <-- AÑADIDA AQUÍ porque feature_order la contiene
        ]

        for col in direct_numeric_cols_from_form:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            else:
                flash(f"Error: Falta el campo numérico requerido '{col}'.", "danger")
                return redirect(url_for('main.index'))
        
        if input_df[direct_numeric_cols_from_form].isnull().any().any():
             nan_cols = input_df[direct_numeric_cols_from_form].isnull().sum()
             error_message_nan = "Error: Campos numéricos con valores no válidos: " + ", ".join(nan_cols[nan_cols > 0].index.tolist())
             flash(error_message_nan, 'danger')
             return redirect(url_for('main.index'))

        # Mapear categóricas
        input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
        input_df['Customer Type'] = input_df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
        input_df['Type of Travel'] = input_df['Type of Travel'].map({'Personal Travel': 0, 'Business travel': 1})
        input_df['Class'] = input_df['flight_class'].map({'Eco': 0, 'Eco Plus': 1, 'Business': 2})
        
        cols_after_mapping = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        for col_am in cols_after_mapping:
            if input_df[col_am].isnull().any():
                # Trata de obtener la clave original del formulario para un mejor mensaje de error
                original_form_key = col_am 
                if col_am == 'Class': original_form_key = 'flight_class'
                flash(f"Error: Valor no válido para el campo '{original_form_key}'.", "danger")
                return redirect(url_for('main.index'))
            input_df[col_am] = input_df[col_am].astype(int)

        if 'flight_class' in input_df.columns:
            input_df = input_df.drop(columns=['flight_class'])
            print("Columna 'flight_class' eliminada después de crear 'Class'.")

        # NO crear 'Total Delay in Minutes' ya que feature_order NO la contiene
        # El log muestra que 'Total Delay in Minutes' no estaba en "Columnas usadas para X"
        if 'Total Delay in Minutes' in input_df.columns: # Si por alguna razón se creó
             input_df = input_df.drop(columns=['Total Delay in Minutes'])
             print("Columna 'Total Delay in Minutes' eliminada porque no se espera por el modelo.")
        print("Columna 'Total Delay in Minutes' NO se creará/usará para el modelo (según feature_order).")


        print("DataFrame después de mapeos:\n", input_df.head().to_string())
        print("Columnas disponibles en input_df ANTES de alinear con feature_order:", input_df.columns.tolist())
        print("Columnas esperadas por el modelo (feature_order):", feature_order)
        
        # Crear el DataFrame final en el orden correcto esperado por el modelo
        input_df_for_model = pd.DataFrame(index=[0], columns=feature_order)
        missing_for_model = []
        for col_fo in feature_order:
            if col_fo in input_df.columns:
                input_df_for_model[col_fo] = input_df[col_fo].values
            else:
                # Esta columna está en feature_order pero no se generó en input_df
                missing_for_model.append(col_fo)
        
        if missing_for_model:
            print(f"ERROR CRÍTICO: Las siguientes columnas están en feature_order pero no se generaron en input_df: {missing_for_model}")
            flash(f"Error de preparación de datos: Faltan columnas {missing_for_model}.", "danger")
            return redirect(url_for('main.index'))
        
        print("\nDataFrame FINAL listo para predicción (input_df_for_model):\n", input_df_for_model.head().to_string())
        print("Tipos finales:\n", input_df_for_model.dtypes)

        if input_df_for_model.isnull().values.any():
            nan_final_check = input_df_for_model.isnull().sum()
            print("ERROR CRÍTICO: NaNs detectados en el DataFrame final ANTES de la predicción.")
            print(nan_final_check[nan_final_check > 0])
            flash('Error interno de datos (NaNs) antes de la predicción. Contacte al administrador.', 'danger')
            return redirect(url_for('main.index'))
            
        print("\n--- Iniciando Predicción ---")
        prediction = model.predict(input_df_for_model)[0]
        prediction_proba = model.predict_proba(input_df_for_model)[0]
        
        probability_satisfied = prediction_proba[1]
        satisfaction_result = "Satisfecho ✅" if prediction == 1 else "Neutral o Insatisfecho ❌"

        print(f"Predicción: {prediction} ({satisfaction_result}), Probabilidad Satisfecho: {probability_satisfied:.4f}")

        # Guardar en DB
        new_entry = NewFlightData(
            customer_name=personal_data.get('customer_name'),
            customer_email=personal_data.get('customer_email'),
            customer_dni=personal_data.get('customer_dni'),
            customer_phone=personal_data.get('customer_phone'),
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

        return render_template('predict.html',
                               prediction_text=satisfaction_result,
                               probability=f"{probability_satisfied:.2%}")

    except ValueError as ve:
        print(f"Error de Valor/Tipo al procesar datos: {ve}")
        print(traceback.format_exc())
        flash(f'Error en los datos ingresados. Detalles: {ve}. Por favor, revisa los campos.', 'danger')
        return redirect(url_for('main.index'))
    except KeyError as ke:
         print(f"Error de Clave: '{ke}'. Posiblemente falta un campo del formulario o error en nombre de columna.")
         print(traceback.format_exc())
         flash(f"Error interno: Falta información requerida ('{ke}'). Por favor, revisa todos los campos.", 'danger')
         return redirect(url_for('main.index'))
    except Exception as e:
        print(f"Error inesperado durante la predicción o guardado: {e}")
        print(traceback.format_exc())
        db.session.rollback()
        flash(f'Ocurrió un error inesperado ({type(e).__name__}) al procesar tu solicitud. Contacte al administrador.', 'danger')
        return redirect(url_for('main.index'))