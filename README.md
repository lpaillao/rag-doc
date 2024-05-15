
# Chatbot con RAG y Embeddings

Este proyecto es una aplicación de chatbot que utiliza Recuperación con Generación Aumentada (RAG) y embeddings para proporcionar respuestas basadas en documentos procesados. El proyecto permite cargar archivos PDF, procesarlos en archivos JSON, generar embeddings y realizar consultas mediante una interfaz de chat moderna.

## Características

- Cargar y procesar archivos PDF
- Generar embeddings a partir de los documentos procesados
- Consultar documentos mediante una interfaz de chat
- Selección de embeddings disponibles para consultas
- Interfaz de usuario con tema oscuro

## Requisitos

- Python 3.8 o superior
- Flask
- Flask-CORS
- PyMuPDF
- Sentence Transformers
- Faiss
- OpenAI API Key

## Instalación

1. Clona este repositorio:

    ```sh
    git clone https://github.com/lpaillao/rag-doc
    cd tu_repositorio
    ```

2. Crea y activa un entorno virtual:

    ```sh
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3. Instala las dependencias:

    ```sh
    pip install -r requirements.txt
    ```

## Configuración

1. **OpenAI API Key**: Obtén tu clave de API de OpenAI y configúrala en el archivo `config.py`:

    ```python
    openai.api_key = 'tu-api-key'
    ```

2. **Archivos de Configuración**: Asegúrate de tener las siguientes estructuras de carpetas en tu directorio `data`:

    ```plaintext
    data/
    ├── raw/
    │   ├── pdf/
    ├── embeddings/
    └── faiss_index/
    ```

## Uso

1. **Ejecutar la Aplicación**:

    ```sh
    python app.py
    ```

2. **Acceder a la Aplicación**: Abre `http://127.0.0.1:5000` en tu navegador para interactuar con la aplicación.

## Despliegue en Heroku

1. **Instalar Heroku CLI**: Descarga e instala la [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).

2. **Crear un archivo `Procfile`**:

    ```plaintext
    web: python app.py
    ```

3. **Crear un archivo `requirements.txt`**:

    ```sh
    pip freeze > requirements.txt
    ```

4. **Iniciar un repositorio Git**:

    ```sh
    git init
    git add .
    git commit -m "Initial commit"
    ```

5. **Desplegar en Heroku**:

    ```sh
    heroku create
    git push heroku master
    heroku ps:scale web=1
    heroku open
    ```

## Estructura del Proyecto

```plaintext
proyecto_chatbot/
├── app.py
├── templates/
│   ├── index.html
│   └── chat.html
├── static/
│   ├── style.css
│   ├── dark-theme.css
├── scripts/
│   ├── pdf_to_json.py
│   ├── generate_embeddings.py
│   ├── query_and_generate.py
├── data/
│   ├── raw/
│   │   ├── pdf/
│   ├── embeddings/
│   └── faiss_index/
├── venv/
├── requirements.txt
├── README.md
```

## Dependencias

- Flask
- Flask-CORS
- PyMuPDF
- Sentence Transformers
- Faiss
- OpenAI

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio que desees realizar.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para obtener más información.
