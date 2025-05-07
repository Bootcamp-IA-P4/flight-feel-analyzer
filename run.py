# run.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Debug=True es útil para desarrollo, Flask usará el valor de FLASK_DEBUG del .env
    # Host='0.0.0.0' permite acceso desde otras máquinas en la red local
    app.run(host='0.0.0.0', port=5000)