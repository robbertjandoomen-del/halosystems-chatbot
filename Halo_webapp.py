import streamlit as st
from pypdf import PdfReader
import openai
import faiss
import numpy as np

# --- Zet je OpenAI API key hier ---
openai.api_key = "YOUR_API_KEY"

# -------------------------------
# Functies
# -------------------------------
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_chunks(chunks):
    vectors = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk, model="text-embedding-ada-002"
        )
        vectors.append(response["data"][0]["embedding"])
    return np.array(vectors).astype("float32")

def build_index(vectors):
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def answer_question(question, chunks, index, top_k=3):
    q_embed = openai.Embedding.create(
        input=question, model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    D, I = index.search(np.array([q_embed], dtype="float32"), top_k)
    relevant_chunks = [chunks[i] for i in I[0]]

    prompt = f"Gebruik onderstaande informatie om de vraag te beantwoorden:\n\n{relevant_chunks}\n\nVraag: {question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# -------------------------------
# Streamlit Web UI
# -------------------------------
st.set_page_config(page_title="HALO Studenten Chatbot", page_icon="📚", layout="wide")

st.title("📚 HALO Studenten Chatbot")
st.markdown("Upload je handleidingen (Brightspace, Osiris, etc.) en stel vragen.")

uploaded_files = st.file_uploader("Upload meerdere PDF’s", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    for f in uploaded_files:
        all_text += load_pdf(f)

    chunks = split_text(all_text)
    vectors = embed_chunks(chunks)
    index = build_index(vectors)

    st.session_state["chunks"] = chunks
    st.session_state["index"] = index
    st.success("Handleidingen succesvol ingeladen ✅")

if "chunks" in st.session_state and "index" in st.session_state:
    question = st.text_input("✏️ Stel je vraag:")
    if question:
        answer = answer_question(question, st.session_state["chunks"], st.session_state["index"])
        st.subheader("💡 Antwoord:")
        st.write(answer)
