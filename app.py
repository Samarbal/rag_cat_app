import streamlit as st
from rag import load_dataset, build_vector_db, retrieve, generate_response

st.set_page_config(page_title="🐱 Cat Facts RAG", page_icon="🐱")
st.title("🐱 Cat Facts Chatbot")
st.caption("Ask me anything about cats — answers are grounded in a dataset of cat facts.")

# Build the vector DB once and cache it across reruns (Streamlit reruns the script on every interaction)
@st.cache_resource(show_spinner="Loading dataset and building embeddings...")
def get_vector_db():
    dataset = load_dataset()
    build_vector_db(dataset)
    return True

get_vector_db()

# Keep chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
query = st.chat_input("Ask a question about cats...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve(query)
            answer = generate_response(query, retrieved)
            st.write(answer)

            with st.expander("🔍 Retrieved context"):
                for chunk, score in retrieved:
                    st.write(f"**({score:.2f})** {chunk}")

    st.session_state.messages.append({"role": "assistant", "content": answer})