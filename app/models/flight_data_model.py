# app/models/flight_data_model.py
from .. import db # Importar db desde __init__.py del paquete padre

class NewFlightData(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # --- Nuevos Datos Personales ---
    customer_name = db.Column(db.String(100), nullable=True) # O False si es requerido
    customer_email = db.Column(db.String(120), nullable=True, unique=False) # unique=True si quieres emails únicos
    customer_dni = db.Column(db.String(20), nullable=True, unique=False) # unique=True si quieres DNI únicos
    customer_phone = db.Column(db.String(20), nullable=True)

    # --- Características de Entrada (como se recibieron del formulario) ---
    gender = db.Column(db.String(10))
    customer_type = db.Column(db.String(20))
    age = db.Column(db.Integer)
    type_of_travel = db.Column(db.String(20))
    flight_class = db.Column(db.String(15)) # Renombrado de 'Class'
    flight_distance = db.Column(db.Integer)
    inflight_wifi_service = db.Column(db.Integer)
    departure_arrival_time_convenient = db.Column(db.Integer)
    ease_of_online_booking = db.Column(db.Integer)
    gate_location = db.Column(db.Integer)
    food_and_drink = db.Column(db.Integer)
    online_boarding = db.Column(db.Integer)
    seat_comfort = db.Column(db.Integer)
    inflight_entertainment = db.Column(db.Integer)
    onboard_service = db.Column(db.Integer) # Renombrado de 'On-board service'
    leg_room_service = db.Column(db.Integer)
    baggage_handling = db.Column(db.Integer)
    checkin_service = db.Column(db.Integer)
    inflight_service = db.Column(db.Integer)
    cleanliness = db.Column(db.Integer)
    departure_delay_in_minutes = db.Column(db.Integer)

    # --- Resultado de la Predicción ---
    predicted_satisfaction = db.Column(db.Integer) # 0 o 1
    prediction_probability_satisfied = db.Column(db.Float) # Probabilidad de ser 1

    def __repr__(self):
        return f'<NewFlightData ID: {self.id}, Name: {self.customer_name}>'