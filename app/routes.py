# app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np # Aún útil para algunos tipos o NaN
import traceback

from . import db
from .model_loader import get_model # Solo necesitamos el modelo (pipeline)
from .models.flight_data_model import NewFlightData

main_bp = Blueprint('main', __name__)

# Cargar el modelo (pipeline completo)
model = get_model()

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/about')
def about():
    return render_template('about-team.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    if not model:
        flash('Error: El modelo de predicción no está cargado. Contacte al administrador.', 'danger')
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

        # --- Preparar datos para el MODELO (PIPELINE) ---
        # El modelo (pipeline) espera un DataFrame.
        # Las columnas deben coincidir con lo que el pipeline espera ANTES de su propio preprocesamiento.
        # Si tu compañera usó 'flight_class' antes de mapearlo a 'Class' en su pipeline,
        # entonces el formulario debe enviar 'flight_class'.
        
        # Crear un DataFrame de una fila. Los nombres de las columnas deben ser los que
        # el pipeline de tu compañera espera en su entrada inicial.
        # Las columnas de retraso ya NO SE ENVÍAN desde el formulario.
        
        # Datos que irán al modelo (excluyendo los personales)
        model_input_data_dict = {k: v for k, v in form_data.items() if k not in personal_data}
        
        # IMPORTANTE: Convertir a los tipos numéricos que el pipeline podría esperar ANTES de su propia lógica.
        # Por ejemplo, si el pipeline espera que 'Age' sea un int, y el form lo manda como string.
        # O si los ratings son strings y el pipeline los espera como int/float.
        # Esto depende de cómo tu compañera haya construido su pipeline.
        # Generalmente, es buena idea convertir los que son inherentemente numéricos.
        
        cols_to_convert_to_numeric = [
            'Age', 'Flight Distance', 'Inflight wifi service',
            'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'
        ]
        
        for col in cols_to_convert_to_numeric:
            if col in model_input_data_dict:
                try:
                    model_input_data_dict[col] = pd.to_numeric(model_input_data_dict[col])
                except ValueError:
                    flash(f"Error: El valor para '{col}' debe ser numérico.", "danger")
                    return redirect(url_for('main.index'))
            # else:
                # Podrías manejar el caso de que una columna esperada falte del form, si es necesario

        input_df_for_pipeline = pd.DataFrame([model_input_data_dict])
        print("DataFrame listo para el pipeline del modelo:\n", input_df_for_pipeline.to_string())


        # --- 3. Realizar Predicción usando el Pipeline Completo ---
        # El pipeline se encarga del preprocesamiento interno.
        # El modelo espera un DataFrame.
        
        # Si el modelo (pipeline) fue entrenado esperando columnas que ahora no envías
        # (como las de retraso), ESTO FALLARÁ. Debes estar seguro de las columnas que su pipeline espera.
        prediction = model.predict(input_df_for_pipeline)[0]
        prediction_proba = model.predict_proba(input_df_for_pipeline)[0]
        probability_satisfied = prediction_proba[1]
        satisfaction_result = "Satisfecho ✅" if prediction == 1 else "Neutral o Insatisfecho ❌"

        print(f"Predicción: {prediction} ({satisfaction_result}), Probabilidad Satisfecho: {probability_satisfied:.4f}")

        # --- 4. Guardar en Base de Datos (Datos ORIGINALES + Predicción) ---
        # Nota: Ya no guardamos 'departure_delay_in_minutes' ni 'arrival_delay_in_minutes'
        new_entry_data = {
            'customer_name': personal_data['customer_name'],
            'customer_email': personal_data['customer_email'],
            'customer_dni': personal_data['customer_dni'],
            'customer_phone': personal_data['customer_phone'],
            'gender': form_data.get('Gender'),
            'customer_type': form_data.get('Customer Type'),
            'age': int(form_data.get('Age', 0)),
            'type_of_travel': form_data.get('Type of Travel'),
            'flight_class': form_data.get('Class'),
            'flight_distance': int(form_data.get('Flight Distance', 0)),
            'inflight_wifi_service': int(form_data.get('Inflight wifi service', 0)),
            'departure_arrival_time_convenient': int(form_data.get('Departure/Arrival time convenient', 0)),
            'ease_of_online_booking': int(form_data.get('Ease of Online booking', 0)),
            'gate_location': int(form_data.get('Gate location', 0)),
            'food_and_drink': int(form_data.get('Food and drink', 0)),
            'online_boarding': int(form_data.get('Online boarding', 0)),
            'seat_comfort': int(form_data.get('Seat comfort', 0)),
            'inflight_entertainment': int(form_data.get('Inflight entertainment', 0)),
            'onboard_service': int(form_data.get('On-board service', 0)),
            'leg_room_service': int(form_data.get('Leg room service', 0)),
            'baggage_handling': int(form_data.get('Baggage handling', 0)),
            'checkin_service': int(form_data.get('Checkin service', 0)),
            'inflight_service': int(form_data.get('Inflight service', 0)),
            'cleanliness': int(form_data.get('Cleanliness', 0)),
            'departure_delay_in_minutes': int(form_data.get('Departure Delay in Minutes', 0)),
            'predicted_satisfaction': int(prediction),
            'prediction_probability_satisfied': float(probability_satisfied),
        }
        new_entry = NewFlightData(**new_entry_data)

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
        flash(f'Error en los datos ingresados ({ve}). Asegúrate de que los números sean válidos y los campos correctos.', 'danger')
        return redirect(url_for('main.index'))
    except KeyError as ke:
         print(f"Error de Clave: Falta una clave o columna esperada: {ke}.")
         print(traceback.format_exc())
         flash(f'Error interno: Falta información requerida. El modelo podría esperar una columna que no se envió. Detalle: {ke}', 'danger')
         return redirect(url_for('main.index'))
    except Exception as e:
        print(f"Error inesperado durante la predicción o guardado: {e}")
        print(traceback.format_exc())
        try:
            db.session.rollback()
            print("Rollback de la sesión de base de datos realizado.")
        except Exception as rb_e:
            print(f"Error adicional durante el rollback: {rb_e}")
        flash(f'Ocurrió un error inesperado. Inténtalo de nuevo o contacta al administrador.', 'danger')
        return redirect(url_for('main.index'))