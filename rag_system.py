import warnings
from sentence_transformers import SentenceTransformer
from wikipediaapi import Wikipedia
import numpy as np
import subprocess
import os

# Suppress warnings (optional)
warnings.filterwarnings("ignore", message="NotOpenSSLWarning")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the sentence embedding model
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)

# Initialize Wikipedia API
wiki = Wikipedia('RAGBot/0.0', 'en')

# Get text from the Studio Ghibli Wikipedia page
doc1 = wiki.page('Studio_Ghibli').text
doc2 = wiki.page('Castle_in_the_Sky').text
doc = doc1 + "\n\n" + doc2

# Split text into manageable paragraphs
paragraphs = doc.split('\n\n')

# Batch encoding function to prevent memory overload
def batch_encode(paragraphs, model, batch_size=5):
    embeddings = []
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        batch_embeddings = model.encode(batch, normalize_embeddings=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Encode each paragraph into embeddings using batch encoding
docs_embed = batch_encode(paragraphs, model)

# Updated query for better clarity
query = "What was the first official Studio Ghibli film?"

# Encode the query into embeddings
query_embed = model.encode([query], normalize_embeddings=True)

# Calculate similarities between the query and paragraphs
similarities = np.dot(docs_embed, query_embed.T)

# Fetch the top 10 most similar paragraphs for richer context
top_10_idx = np.argsort(similarities, axis=0)[-10:].flatten().tolist()
most_similar_documents = [paragraphs[idx] for idx in top_10_idx]

# Manually add introductory context
additional_context = (
    "Studio Ghibli was officially founded on June 15, 1985. "
    "Their first official film is widely recognized as 'Castle in the Sky,' released in 1986. "
    "Below is more information about the studio's early works:\n\n"
)
CONTEXT = additional_context + "\n\n".join(most_similar_documents)

# Construct the prompt for Llama3
prompt_llama3 = f"""
Please identify the first official Studio Ghibli film based on the information provided below.

{CONTEXT}

What was the first official Studio Ghibli film?

If you don't know the answer, respond with "I don't have enough information."
"""

# Construct the prompt for Mistral with slight adjustments
prompt_mistral = f"""
Based on the following information, what was the first official Studio Ghibli film?

{CONTEXT}

Please provide the answer, or say "I don't have enough information" if it isn't clear from the context.
"""

# Function to run the local LLM (Llama3 or Mistral) using Ollama
def run_local_llm(model_name, prompt):
    try:
        # Use subprocess to run the command with input directly passed
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt,
            text=True,
            capture_output=True
        )

        # Check if the command ran successfully
        if result.returncode != 0:
            raise RuntimeError(f"Error running {model_name}: {result.stderr}")

        return result.stdout
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Run the Llama3 model with the refined prompt
llama3_response = run_local_llm("llama3", prompt_llama3)
print(f"Llama3 Response:\n{llama3_response}")

# Run the Mistral model with the adjusted prompt
mistral_response = run_local_llm("mistral", prompt_mistral)
print(f"Mistral Response:\n{mistral_response}")
