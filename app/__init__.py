# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
print(f"Buscando .env en: {dotenv_path}")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(".env cargado.")
else:
    print("Advertencia: archivo .env no encontrado.")

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key-please-change')

    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    db_host = os.environ.get('DB_HOST')
    db_name = os.environ.get('DB_NAME')

    if not all([db_user, db_password, db_host, db_name]):
        print("Error Crítico: Faltan variables de entorno para la base de datos. Usando SQLite como fallback.")
        basedir = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(basedir, 'database', 'predictions.db')
        os.makedirs(os.path.join(basedir, 'database'), exist_ok=True)
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
        print(f"Configurando base de datos MySQL: mysql+mysqlconnector://{db_user}:***@{db_host}/{db_name}")

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    # --- Registrar el Blueprint ---
    from .routes import main_bp # Importa el blueprint
    app.register_blueprint(main_bp) # Registra el blueprint en la app

    # Crear tablas
    with app.app_context():
        # Importa modelos aquí dentro para evitar problemas de importación circular
        # y asegurar que el contexto de la app esté activo
        from .models.flight_data_model import NewFlightData
        print("Verificando/creando tablas de base de datos...")
        try:
            db.create_all()
            print("Tablas verificadas/creadas.")
        except Exception as e:
             print(f"Error al crear las tablas de la base de datos: {e}")
             print("Asegúrate de que la base de datos exista y las credenciales en .env sean correctas.")

    print("Aplicación Flask creada.")
    return app