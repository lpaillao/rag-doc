import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_preprocessed_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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

def update_faiss_index(embeddings, index_file_path):
    index = faiss.read_index(index_file_path)
    index.add(embeddings)
    faiss.write_index(index, index_file_path)

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Uso: python generate_embeddings.py <nombre_del_archivo_json> <nombre_del_archivo_embeddings> [nombre_del_archivo_index]")
        sys.exit(1)

    json_filename = sys.argv[1]
    embeddings_filename = sys.argv[2]
    index_filename = sys.argv[3] if len(sys.argv) == 4 else None

    # Definir las rutas
    json_file_path = os.path.join('data', 'raw', json_filename)
    embeddings_file_path = os.path.join('data', embeddings_filename)
    index_file_path = os.path.join('data', 'faiss_index.bin') if index_filename is None else os.path.join('data', index_filename)

    # Cargar documentos preprocesados
    documents = load_preprocessed_documents(json_file_path)
    
    # Generar embeddings
    embeddings = generate_embeddings(documents)
    
    # Guardar embeddings
    save_embeddings(embeddings, embeddings_file_path)
    
    # Crear o actualizar índice FAISS
    if index_filename:
        update_faiss_index(embeddings, index_file_path)
    else:
        create_faiss_index(embeddings, index_file_path)
    
    print("Embeddings generated and FAISS index created/updated.")
