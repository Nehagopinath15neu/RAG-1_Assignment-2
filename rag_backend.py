import requests
import json
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
import numpy as np
import torch

# Force the model to run on CPU (avoiding MPS memory issues)
device = torch.device("cpu")
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True).to(device)
wiki = Wikipedia('RAGBot/0.0', 'en')

# Function to interact with the LLM API
def call_llm(model_name, user_input, temperature=0.7, max_tokens=100):
    url = "http://localhost:11434/api/chat"  # Ollama API URL
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": user_input.strip()}
        ]
    }
    response = requests.post(url, json=payload, stream=True)
    full_response = ""
    for chunk in response.iter_lines():
        if chunk:
            chunk_data = json.loads(chunk.decode("utf-8"))
            full_response += chunk_data.get("message", {}).get("content", "")
            if chunk_data.get("done", False):
                break
    return full_response

# RAG system function to retrieve relevant Wikipedia content dynamically
def rag_system(query):
    # Dynamically search for the relevant Wikipedia page based on the query
    page = wiki.page(query)

    if not page.exists():
        return f"I couldn't find any relevant information about '{query}' on Wikipedia."

    # Split the page content into paragraphs
    paragraphs = page.text.split('\n\n')

    # Encode the paragraphs and the query
    docs_embed = batch_encode(paragraphs, model)
    query_embed = model.encode([query], normalize_embeddings=True)

    # Find the top 5 most relevant paragraphs
    similarities = np.dot(docs_embed, query_embed.T)
    top_5_idx = np.argsort(similarities, axis=0)[-5:].flatten().tolist()
    most_similar_documents = [paragraphs[idx] for idx in top_5_idx]

    # Construct the context with relevant paragraphs
    context = "\n\n".join(most_similar_documents)
    return f"Context:\n{context}\n\nQuestion: {query}"

# Helper function to batch encode paragraphs to avoid memory issues
def batch_encode(paragraphs, model, batch_size=2):  # Reduced batch size
    embeddings = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        batch_embeddings = model.encode(batch, normalize_embeddings=True, device="cpu")
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Function to process user input and route queries through RAG or Non-RAG mode
def get_llm_response(model_name, user_query, use_rag=False):
    if use_rag:
        response = rag_system(user_query)
        # Fallback to direct LLM query if RAG mode fails to find relevant content
        if "I couldn't find any relevant information" in response:
            return call_llm(model_name, user_query)
        return response
    else:
        return call_llm(model_name, user_query)
