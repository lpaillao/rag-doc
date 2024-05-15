from flask import Flask, request, render_template, redirect, url_for, jsonify
from flask_cors import CORS
import os
import json
import fitz  # PyMuPDF
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
import config  # Importar el archivo de configuración

app = Flask(__name__)
CORS(app)

# Configuración de la API de OpenAI
openai.api_key = config.OPENAI_API_KEY

# Rutas de carpetas
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['JSON_FOLDER'] = config.JSON_FOLDER
app.config['EMBEDDINGS_FOLDER'] = config.EMBEDDINGS_FOLDER

def clean_text(text):
    # Eliminar caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios adicionales
    text = text.strip()
    return text

def pdf_to_json(pdf_path, json_output_path):
    try:
        # Abrir el archivo PDF
        document = fitz.open(pdf_path)
        pages_data = []
        # Extraer texto de cada página
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text("text")
            cleaned_text = clean_text(text)
            page_data = {
                "page_number": page_num + 1,
                "original_text": text,
                "cleaned_text": cleaned_text
            }
            pages_data.append(page_data)
        # Guardar los datos extraídos en un archivo JSON
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, ensure_ascii=False, indent=4)
        return pages_data
    except Exception as e:
        print(f"Error al procesar el archivo PDF: {e}")
        return None

def generate_embeddings(documents, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    texts = [doc['cleaned_text'] for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def create_faiss_index(embeddings, file_path):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, file_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        json_filename = filename.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
        file.save(pdf_path)
        pages_data = pdf_to_json(pdf_path, json_path)
        return jsonify({"status": "success", "json_path": json_path, "pages_data": pages_data})

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings_route():
    data = request.get_json()
    original_json_path = data['json_path']
    embeddings_name = data['embeddings_name']
    
    # Renombrar y mover el archivo JSON
    new_json_path = os.path.join(app.config['JSON_FOLDER'], embeddings_name + '.json')
    os.rename(original_json_path, new_json_path)

    with open(new_json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    embeddings = generate_embeddings(documents)
    embeddings_file_path = os.path.join(app.config['EMBEDDINGS_FOLDER'], embeddings_name + '.npy')
    index_file_path = os.path.join(app.config['EMBEDDINGS_FOLDER'], embeddings_name + '_faiss_index.bin')
    
    save_embeddings(embeddings, embeddings_file_path)
    create_faiss_index(embeddings, index_file_path)
    
    return jsonify({"status": "success", "embeddings_file_path": embeddings_file_path, "index_file_path": index_file_path})

@app.route('/embeddings')
def list_embeddings():
    files = os.listdir(app.config['EMBEDDINGS_FOLDER'])
    embeddings_files = [f for f in files if f.endswith('.npy')]
    embeddings_files = [f.rsplit('.', 1)[0] for f in embeddings_files]  # Remove file extension
    return jsonify(embeddings_files)

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data['query']
    embeddings_name = data['embeddings_name']
    json_path = os.path.join(app.config['JSON_FOLDER'], embeddings_name + '.json')
    embeddings_path = os.path.join(app.config['EMBEDDINGS_FOLDER'], embeddings_name + '.npy')
    index_path = os.path.join(app.config['EMBEDDINGS_FOLDER'], embeddings_name + '_faiss_index.bin')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    embeddings = np.load(embeddings_path)
    index = faiss.read_index(index_path)
    
    model = SentenceTransformer('all-mpnet-base-v2')
    relevant_docs = search_documents(query, model, index, documents)
    
    response = generate_response(query, relevant_docs)
    
    return jsonify({
        "question": query,
        "response": response['choices'][0]['message']['content'].strip(),
        "total_tokens": response['usage']['total_tokens'],
        "prompt_tokens": response['usage']['prompt_tokens'],
        "completion_tokens": response['usage']['completion_tokens'],
        "context": [doc['original_text'] for doc in relevant_docs]
    })

def search_documents(query, model, index, documents, k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k)
    return [documents[i] for i in I[0]]

def generate_response(query, relevant_docs):
    context = " ".join([doc['original_text'] for doc in relevant_docs])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {query}"}
        ],
        max_tokens=150
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
