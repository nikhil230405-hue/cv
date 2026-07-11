import streamlit as st
import PyPDF2
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from groq import Groq

# ----------------------------------------------
# Page Configuration
# ----------------------------------------------
st.set_page_config(
    page_title="University FAQ Assistant",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------
# Sidebar
# ----------------------------------------------
with st.sidebar:
    st.title("📚 University FAQ Assistant")
    st.markdown("""
    ### How to Use:
    1. Upload the university FAQ PDF  
    2. Ask any question  
    3. Model retrieves relevant content  
    4. LLM generates answer  
    """)
    st.markdown("---")

# ----------------------------------------------
# Load API Key
# ----------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ----------------------------------------------
# Load Classification Model
# ----------------------------------------------
model = pickle.load(open("question_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def classify_question(question):
    vector = tfidf.transform([question])
    prediction = model.predict(vector)[0]
    return prediction

# ----------------------------------------------
# PDF TEXT EXTRACTION
# ----------------------------------------------
def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ----------------------------------------------
# CHUNKING
# ----------------------------------------------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ----------------------------------------------
# EMBEDDINGS
# ----------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return np.array(embedder.encode(texts)).astype("float32")

# ----------------------------------------------
# SEARCH Using NearestNeighbors
# ----------------------------------------------
def build_nn_index(embeddings):
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(embeddings)
    return nn

def search_nn(query, chunks, chunk_emb, nn_index):
    q_emb = embedder.encode([query])
    distances, indices = nn_index.kneighbors(q_emb)
    return [chunks[i] for i in indices[0]]

# ----------------------------------------------
# LLM Response
# ----------------------------------------------
def groq_answer(question, context):
    prompt = f"""
Use ONLY this university context to answer:

{context}

Question: {question}
"""
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250,
        temperature=0
    )
    return res.choices[0].message.content

# ----------------------------------------------
# MAIN UI
# ----------------------------------------------
st.markdown("""
<div style="text-align:center;">
    <h1>🎓 University FAQ Assitant</h1>
    <p style="font-size:17px; color:gray;">
        Ask any question related to university admissions,fees, hostels, courses and more.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# PDF Upload
pdf = st.file_uploader("📄 **Upload University FAQ PDF**", type="pdf")

if pdf:
    st.success("✅ PDF uploaded successfully!")

    text = extract_pdf_text(pdf)
    chunks = chunk_text(text)
    chunk_emb = get_embeddings(chunks)
    nn_index = build_nn_index(chunk_emb)

    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Type your question here...")

    if question:
        q_type = classify_question(question)
        st.info(f"📌 **Predicted Question Type:** `{q_type}`")

        retrieved = search_nn(question, chunks, chunk_emb, nn_index)
        context = "\n\n".join(retrieved)

        answer = groq_answer(question, context)

        st.markdown("### 🟦 Chatbot Answer")
        st.markdown(f"""
        <div style="
            padding:18px; 
            border-radius:10px; 
            background:#000000; 
            color:white; 
            font-size:16px; 
            line-height:1.6;">
        {answer}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔍 Retrieved Relevant Chunks")
        for i, c in enumerate(retrieved):
            with st.expander(f"Chunk {i+1}"):
                st.write(c)

else:
    st.warning("⬆️ Please upload a FAQ PDF to begin.")
