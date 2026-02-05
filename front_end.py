import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📄 RAG PDF Chatbot")



try:
    health = requests.get(f"{API_BASE}/health").json()
    st.success("Backend Connected")
except:
    st.error("Backend not running")
    st.stop()

# ----------------------------
# Upload PDF
# ----------------------------

st.header("Upload PDF")

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file:

    if st.button("Upload & Process"):

        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        }

        with st.spinner("Processing PDF..."):
            response = requests.post(f"{API_BASE}/upload-pdf", files=files)

        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(response.json()["detail"])

# ----------------------------
# Chat Section
# ----------------------------

st.header("Ask Questions")

question = st.text_input("Enter your question:")

if st.button("Ask"):

    payload = {
        "question": question
    }

    with st.spinner("Thinking..."):
        response = requests.post(f"{API_BASE}/ask", json=payload)

    if response.status_code == 200:
        data = response.json()

        st.subheader("Answer:")
        st.write(data["answer"])

    else:
        st.error(response.json()["detail"])

# ----------------------------
# Chat History
# ----------------------------

if st.button("Show Chat History"):

    response = requests.get(f"{API_BASE}/chat-history")

    if response.status_code == 200:
        history = response.json()["chat_history"]

        st.subheader("Conversation")

        for msg in history:
            if msg["type"] == "human":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

    else:
        st.error("Failed to load chat history")
