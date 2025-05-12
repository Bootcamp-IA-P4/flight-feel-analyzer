# Dockerfile (en la raíz de flight-feel-analyzer/)

# 1. Usar una imagen base oficial de Python
FROM python:3.13-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app_root

# 3. Copiar el archivo de requerimientos primero
COPY requirements.txt .

# 4. Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO el contenido del directorio actual (raíz del proyecto)
# al directorio de trabajo /app_root del contenedor.
# Esto incluye la carpeta 'app', 'run.py', etc.
COPY . .

# 6. Exponer el puerto
EXPOSE 5000

# 7. Comando para ejecutar la aplicación
# FLASK_APP está seteado a 'app' en docker-compose.yml (refiriéndose al paquete app/)
# Este comando le dice a Flask que ejecute la app encontrada en ese paquete,
# buscando una función create_app() o una instancia 'app'.
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]