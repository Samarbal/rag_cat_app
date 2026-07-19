import os
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from openai import OpenAI

#  set up 
load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, well-supported
GENERATION_MODEL = "llama-3.3-70b-versatile"  

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))


deepseek_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1" 
)


#  ****  1. load dataset 
def load_dataset (path= 'cat-facts.txt'): 
    with open(path, 'r', encoding ='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f'Loaded {len(lines)} facts')
    return lines

   

# ***** 2. Embeddings 
def get_embedding(text):
    vector = hf_client.feature_extraction(text, model = EMBEDDING_MODEL)
    vector = np.array(vector)

     # some models return per-token vectors (2D) instead of one pooled vector -> average them
    if vector.ndim > 1:
        vector = vector.mean(axis=0)

    return vector 

VECTOR_DB = []  # list of (chunk, embedding) tuples

def build_vector_db (chunks):
    global VECTOR_DB
    VECTOR_DB = []
    for i, chunk in enumerate (chunks):
        embedding = get_embedding(chunk)
        VECTOR_DB.append((chunk, embedding))
        print (f"Embedded {i+1} / {len(chunks)}")

    return VECTOR_DB

# **** 3. Retrieval 
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve (query , top_n = 3):
    query_embedding = get_embedding(query)
    scored = [(chunk, cosine_similarity(query_embedding, emb)) for chunk, emb in VECTOR_DB]
    scored.sort(key = lambda x:x[1], reverse= True)
    return scored[:top_n]


#  ****  4. response generation 
def generate_response (query, retrieved_chunks):
    context = "\n".join([f" - {chunk}" for chunk, _ in retrieved_chunks])
    
    instruction_prompt = f"""You are a helpful chatbot that only knows about cats.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context}
"""
    
    response = deepseek_client.chat.completions.create(
        model=GENERATION_MODEL, 

        messages=[
            {"role":"system", "content": instruction_prompt}, 
            {"role":"user", "content": query}

        ], 
    )

    return response.choices[0].message.content


#  Quick test
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