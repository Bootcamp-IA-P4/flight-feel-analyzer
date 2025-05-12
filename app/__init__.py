# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

# Solo intenta cargar .env si la variable de entorno RUNNING_IN_DOCKER_COMPOSE no es 'true'
# Esta variable la estableces en docker-compose.yml
RUNNING_IN_DOCKER_COMPOSE = os.environ.get('RUNNING_IN_DOCKER_COMPOSE', 'false').lower() == 'true'

if not RUNNING_IN_DOCKER_COMPOSE:
    # Construir la ruta al .env en la raíz del proyecto (un nivel arriba de la carpeta 'app')
    project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root_dir, '.env')
    print(f"No ejecutando en Docker Compose (o RUNNING_IN_DOCKER_COMPOSE no es true). Buscando .env en: {dotenv_path}")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(".env cargado por la aplicación.")
    else:
        print(f"Advertencia: .env no encontrado en {dotenv_path} para carga local.")
else:
    print("Ejecutando en Docker Compose. Se esperan variables de entorno directamente (ignorar búsqueda local de .env).")

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    print("Creando aplicación Flask...")

    # SECRET_KEY
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
    if not app.config['SECRET_KEY']:
        print("ADVERTENCIA: SECRET_KEY no encontrada en el entorno. Usando valor por defecto (¡inseguro!).")
        app.config['SECRET_KEY'] = 'dev-secret-key-please-change-in-env'

    # Configuración de la Base de Datos
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    db_host_from_env = os.environ.get('DB_HOST')
    db_name = os.environ.get('DB_NAME')

    print(f"Variables DB leídas del entorno: USER={db_user}, HOST='{db_host_from_env}', NAME={db_name}")

    if not all([db_user, db_password, db_host_from_env, db_name]):
        print("Error Crítico: Faltan variables de entorno para la base de datos (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME).")
        raise EnvironmentError("Configuración de base de datos incompleta en variables de entorno. Revise su .env y docker-compose.yml")
    else:
        db_port = os.environ.get('DB_PORT', 5432)
        app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_password}@{db_host_from_env}:{db_port}/{db_name}'


    db.init_app(app)

    from .routes import main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        from .models.flight_data_model import NewFlightData
        print("Intentando crear/verificar tablas de base de datos...")
        try:
            db.create_all()
            print("Tablas verificadas/creadas.")
        except Exception as e:
             print(f"ERROR al crear las tablas de la base de datos: {e}")
             print(f"URI usada: {app.config.get('SQLALCHEMY_DATABASE_URI')}")
             print(f"Asegúrate de que la base de datos '{db_name}' exista, las credenciales sean correctas y el host '{db_host_from_env}' sea accesible.")

    print("Aplicación Flask creada y configurada.")
    return app