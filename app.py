import streamlit as st
from rag import load_dataset, build_vector_db, retrieve, generate_response, VECTOR_DB

st.set_page_config(page_title="🐱 Cat Facts RAG", page_icon="🐱")
st.title("🐱 Cat Facts Chatbot")
st.caption("Ask me anything about cats — answers are grounded in a dataset of cat facts.")


@st.cache_resource(show_spinner="Loading dataset and building embeddings...")
def rebuild_cat_vector_db():
    dataset = load_dataset()
    build_vector_db(dataset)
    return True


rebuild_cat_vector_db()

# Sidebar: dataset info + manual rebuild (useful after you edit cat-facts.txt / re-scrape Wikipedia)
with st.sidebar:
    st.subheader("Database")
    st.write(f"**{len(VECTOR_DB)}** facts currently indexed")
    if st.button("🔄 Rebuild database"):
        build_cat_vector_db.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about cats...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retrieved = retrieve(query)
                answer = generate_response(query, retrieved)
            except Exception as e:
                retrieved = []
                answer = f"⚠️ Something went wrong: {e}"

            st.write(answer)

            if retrieved:
                with st.expander(f"🔍 Retrieved context ({len(retrieved)} chunks)"):
                    for chunk, score in retrieved:
                        st.write(f"**({score:.2f})** {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": answer})