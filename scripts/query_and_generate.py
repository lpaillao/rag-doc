import os
import json
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer

# Configuración de la API de OpenAI
openai.api_key = 'tu-api-key'

def load_preprocessed_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_embeddings(file_path):
    return np.load(file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Uso: python query_and_generate.py <json_file> <embeddings_file> <index_file>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    embeddings_file = sys.argv[2]
    index_file = sys.argv[3]

    documents = load_preprocessed_documents(json_file)
    embeddings = load_embeddings(embeddings_file)
    index = load_faiss_index(index_file)

    model = SentenceTransformer('all-mpnet-base-v2')

    while True:
        query = input("Por favor, ingresa tu pregunta (o 'salir' para terminar): ")
        if query.lower() == 'salir':
            break

        relevant_docs = search_documents(query, model, index, documents)
        response = generate_response(query, relevant_docs)

        print("\nPregunta:", query)
        print("Respuesta:", response['choices'][0]['message']['content'].strip())
        print("\nDetalles de la respuesta:")
        print("Tokens utilizados:", response['usage']['total_tokens'])
        print("Tokens de entrada:", response['usage']['prompt_tokens'])
        print("Tokens de salida:", response['usage']['completion_tokens'])
        print("\nContexto enviado a la API:")
        for doc in relevant_docs:
            print("-" * 80)
            print(doc['original_text'])
        print("=" * 80)
