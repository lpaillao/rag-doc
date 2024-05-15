# config.py

import os

# Configuración de la API de OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'tu_api_key_por_defecto')

# Rutas de carpetas
UPLOAD_FOLDER = 'data/raw/pdf'
JSON_FOLDER = 'data/raw'
EMBEDDINGS_FOLDER = 'data'
