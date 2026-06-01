import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AAU AI Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AAU AI Chat Assistant (FAISS)")
st.caption("Semantic FAQ system powered by Sentence-BERT + FAISS")

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(
    "all-MiniLM-L6-v2",
    cache_folder="./models"
)

model = load_model()

# -----------------------------
# 3. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    with open("faq_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

faq_data = load_data()

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# -----------------------------
# 4. EMBEDDINGS
# -----------------------------
@st.cache_resource
def build_embeddings():
    return model.encode(questions, normalize_embeddings=True)

embeddings = build_embeddings()

# -----------------------------
# 5. FAISS INDEX
# -----------------------------
@st.cache_resource
def build_index():
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    return index

index = build_index()

# -----------------------------
# 6. SEARCH FUNCTION
# -----------------------------
def search(query, top_k=3):
    query_vec = model.encode([query], normalize_embeddings=True)

    scores, indices = index.search(np.array(query_vec), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })

    return results

# -----------------------------
# 7. CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 8. DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 9. USER INPUT (CHAT STYLE)
# -----------------------------
user_input = st.chat_input("Ask anything about AAU...")

if user_input:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------
    # FAISS SEARCH
    # -------------------------
    results = search(user_input)
    best = results[0]

    if best["score"] < 0.35:
        response = "❌ Sorry, I couldn't find a relevant answer. Please rephrase your question."
    else:
        response = best["answer"]

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(response)

        # Show top matches (debug / transparency)
        with st.expander("🔍 Top Matching FAQs"):
            for r in results:
                st.markdown(f"**Q:** {r['question']}")
                st.markdown(f"**Score:** {r['score']:.2f}")
                st.markdown("---")
