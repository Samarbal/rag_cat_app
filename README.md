# 🐱 Cat Facts Chatbot: Dynamic RAG Pipeline with Automated Scraping

An advanced, context-grounded AI Chatbot built using the **Retrieval-Augmented Generation (RAG)** architecture. This project showcases a complete production-grade pipeline: it dynamically scrapes medical, behavioral, and historical cat data from Wikipedia, structures a local vector database with over 1,200+ clean knowledge chunks, and utilizes the **Llama-3.3-70b** model via Groq to deliver precise, hallucination-free responses.

---

## The Process Flow
The system logic flows naturally across these 6 automated layers:
```text
[MediaWiki API Scraper] ➔ [Load Datasets] ➔ [Hugging Face Embeddings] ➔ 
[Store In Memory Vector DB] ➔ [Retrieve via Cosine Similarity] ➔ [Grounded LLM Answer]
```

--- 

## Key Features & Evidence

### 1. On-Demand Web Knowledge Harvesting (Offline-Ready)
Instead of straining Wikipedia with live scraping during every single query, our dedicated `scraper.py` script runs on-demand to fetch pure plain-text data once. It then saves it locally, keeping the RAG pipeline fast, efficient, and offline-ready.

<img width="638" height="250" alt="image" src="https://github.com/user-attachments/assets/ed69f328-c675-40f1-8f95-d0dc81a104d8" />


### 2. Semantic Search & Rate-Limited Embedding Loop

The pipeline converts text chunks into dense, multidimensional vectors via Hugging Face's `all-MiniLM-L6-v2`. To ensure production-grade stability over a large dataset (1,250+ chunks), the loop implements a custom cooling delay (`time.sleep`) and an automated retry mechanism (`max_retries=5`) to cleanly bypass Hugging Face's temporary `500 Internal Server Errors` without losing data.

<img width="1079" height="564" alt="image" src="https://github.com/user-attachments/assets/7df4266f-6abd-41df-b919-c6b367baa9df" />


### 3.  Anti-Hallucination Guardrails & Interactive UI

The Streamlit frontend delivers a responsive chat experience featuring continuous session-state memory. To guarantee complete truthfulness, the system prompt enforces strict anti-hallucination boundaries, forcing the **Llama-3.3-70b** model to gracefully refuse or restrict answers if no matching evidence is found. The interactive expander component acts as a live debugger, displaying the exact retrieved text chunks alongside their **Cosine Similarity** scores.

<img width="672" height="805" alt="image" src="https://github.com/user-attachments/assets/91ef33ef-277e-4478-ac16-a4ecee793440" />


---

##  Tech Stack & Dependencies
- **Frontend / UI:** Streamlit
- **LLM Provider:** Groq Cloud Platform (`llama-3.3-70b-versatile`)
- **Embedding Model:** Hugging Face Hub (`sentence-transformers/all-MiniLM-L6-v2`)
- **Networking & API:** `requests` & MediaWiki API
- **Mathematics & Vector Logic:** `numpy` (Dot product & linear algebra norms)

--- 

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd rag_cat_app
   ```

2. **Set up Virtual Environment & Requirements:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   
   # Install frozen structural dependencies
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project root directory:
   ```env
   HF_TOKEN=your_huggingface_secret_token_here
   GROQ_API_KEY=your_groq_cloud_api_key_here
   ```

---

## Execution Guide

1. **Harvest the dynamic web data:**
   ```bash
   python scraper.py
   ```
2. **Launch the responsive web application:**
   ```bash
   streamlit run app.py
   ```


--- 

## License

**Samar M. Balousha** 

