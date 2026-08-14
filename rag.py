

import os
import time
import re
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI
 
# Setup 
load_dotenv()
 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, well-supported
GENERATION_MODEL = "llama-3.3-70b-versatile"
 
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
 
# Renamed for clarity: this is a Groq client, not DeepSeek (kept your GROQ_API_KEY / base_url)
llm_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
 
 
# 1. Load dataset

def load_dataset():
    """Load and clean facts from both the hand-written and Wikipedia-scraped files."""
    chunks = []
    paths = ["cat-facts.txt", "wiki-cat-facts.txt"]
 
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
 
                # Strip Wikipedia section headers like "== History =="
                content = re.sub(r"={2,}.*?={2,}", "", content)
 
                # Split into sentences
                sentences = re.split(r"(?<=[.!?])\s+", content)
 
                for s in sentences:
                    s = s.strip()
                    if s and len(s) > 25:
                        chunks.append(s)
 
        except FileNotFoundError:
            print(f" Warning: file not found: {path}")
 
    # Remove exact duplicates while preserving order (Wikipedia text often repeats facts)
    seen = set()
    unique_chunks = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique_chunks.append(c)
 
    print(f"Loaded {len(unique_chunks)} unique factual chunks total "
          f"({len(chunks) - len(unique_chunks)} duplicates removed)")
    return unique_chunks
 
 
#  2. Embeddings
def get_embedding(text):
    vector = hf_client.feature_extraction(text, model=EMBEDDING_MODEL)
    vector = np.array(vector)
    if vector.ndim > 1:
        vector = vector.mean(axis=0)
    return vector
 
 
VECTOR_DB = []  # list of (chunk, embedding) tuples
 
 
import time

def build_vector_db(chunks, max_retries=5):
    """
    Embed every chunk and store it in VECTOR_DB cleanly.
    Each chunk gets its own retry loop without doubling data.
    """
    global VECTOR_DB
    VECTOR_DB = []
    skipped = 0  
    print(f"Starting to embed {len(chunks)} chunks... Please watch the terminal.")
 
    for i, chunk in enumerate(chunks):
        embedding = None
 
        for attempt in range(max_retries):
            try:
                embedding = get_embedding(chunk)
                break  # Success -> We exit the attempt loop without adding a duplicate.
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ Skipping chunk {i+1} after {max_retries} failed attempts: {e}")
                else:
                    print(f"🔄 Server busy at chunk {i+1}, retrying in 2s...")
                    time.sleep(2)
 
        # The correct and only addition takes place here, outside the attempts loop.
        if embedding is not None:
            VECTOR_DB.append((chunk, embedding))
        else:
            skipped += 1
 
        time.sleep(0.05)  # gentle rate limiting
 
        # Print the correct progress every 20 sentences.
        if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
            print(f"🧬 Progress: embedded {i+1}/{len(chunks)} (skipped so far: {skipped})")
 
    print(f" Done. {len(VECTOR_DB)} chunks embedded, {skipped} skipped.")
    return VECTOR_DB

 
# 3. Retrieval
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 
 
def retrieve(query, top_n=5, min_score=0.2):
    """Return the top_n most relevant chunks above a minimum similarity threshold."""
    query_embedding = get_embedding(query)
    scored = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
    scored.sort(key=lambda x: x[1], reverse=True)
    filtered = [pair for pair in scored if pair[1] >= min_score]
    return filtered[:top_n] if filtered else scored[:top_n]  # fall back if nothing clears the bar
 
 
# ---------- 4. Response generation ----------
def generate_response(query, retrieved_chunks):
    if not retrieved_chunks:
        return "I don't have any information about that in my cat facts database."
 
    context = "\n".join([f"- {chunk}" for chunk, _ in retrieved_chunks])
 
    instruction_prompt = f"""You are a helpful assistant that answers questions about cats.
 
Rules:
- Use ONLY the facts listed below. Do not invent or assume anything not stated.
- If the facts don't fully answer the question, say what you can and note what's missing.
- Write a clear, well-organized answer (2-5 sentences, or a short list if multiple points apply).
- Do not mention "the context" or "the facts provided" in your answer — just answer naturally.
 
Facts:
{context}
"""
 
    try:
        response = llm_client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"
 
 
# Quick test
if __name__ == "__main__":
    dataset = load_dataset()
    build_vector_db(dataset)
 
    query = "tell me about cat speed"
    retrieved = retrieve(query)
 
    print("\nRetrieved chunks:")
    for chunk, score in retrieved:
        print(f" - (similarity: {score:.2f}) {chunk}")
 
    print("\nChatbot response:")
    print(generate_response(query, retrieved))
 
