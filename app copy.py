#!/usr/bin/env python3
"""
AAU Semantic FAQ Chatbot
Powered by Sentence-BERT and FAISS
"""

import streamlit as st
import json
import numpy as np
import faiss
import time
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

# -----------------------------
# Configuration
# -----------------------------
CONFIDENCE_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.45,
    "low": 0.35
}

MODEL_NAME = "paraphrase-MiniLM-L3-v2"  # Faster download than all-MiniLM-L6-v2

# -----------------------------
# 1. Load Model with Error Handling
# -----------------------------
@st.cache_resource
def load_model():
    """Load the Sentence-BERT model with proper error handling"""
    try:
        # Set Hugging Face mirror for faster downloads (optional)
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
        
        with st.spinner(f"Loading AI model... This may take a few minutes on first run."):
            model = SentenceTransformer(MODEL_NAME)
            st.success("✅ Model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please check your internet connection and try again.")
        st.stop()

# -----------------------------
# 2. Load FAQ Data
# -----------------------------
@st.cache_data
def load_faq_data(filepath: str = "faq_data.json") -> Tuple[List[str], List[str]]:
    """Load FAQ questions and answers from JSON file"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            faqs = data
        elif isinstance(data, dict) and "faqs" in data:
            faqs = data["faqs"]
        else:
            st.error("Invalid FAQ data format!")
            st.stop()
        
        questions = [item["question"] for item in faqs]
        answers = [item["answer"] for item in faqs]
        
        st.success(f"✅ Loaded {len(questions)} FAQ entries")
        return questions, answers
    except FileNotFoundError:
        st.error(f"❌ FAQ data file '{filepath}' not found!")
        st.info("Please ensure faq_data.json is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading FAQ data: {e}")
        st.stop()

# -----------------------------
# 3. Build Question Embeddings
# -----------------------------
@st.cache_resource
def build_embeddings(model, questions: List[str]) -> np.ndarray:
    """Create embeddings for all FAQ questions"""
    with st.spinner("Building question embeddings..."):
        embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)
        return embeddings

# -----------------------------
# 4. Build FAISS Index
# -----------------------------
@st.cache_resource
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Create FAISS index for fast similarity search"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings.astype('float32'))
    return index

# -----------------------------
# 5. Search Function
# -----------------------------
def search(query: str, model, index: faiss.Index, questions: List[str], 
           answers: List[str], top_k: int = 3) -> List[Dict]:
    """Search for similar questions and return answers"""
    if not query.strip():
        return []
    
    # Encode the query
    query_vec = model.encode([query], normalize_embeddings=True)
    
    # Search in FAISS index
    scores, indices = index.search(query_vec.astype('float32'), top_k)
    
    # Prepare results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(questions):
            results.append({
                "question": questions[idx],
                "answer": answers[idx],
                "score": float(score),
                "confidence": get_confidence_level(float(score))
            })
    
    return results

def get_confidence_level(score: float) -> str:
    """Determine confidence level based on similarity score"""
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    elif score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    elif score >= CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    else:
        return "very_low"

def format_response(results: List[Dict]) -> Tuple[str, str]:
    """Format the response based on search results"""
    if not results:
        return "I couldn't find an answer to your question. Please try rephrasing or contact the AAU Registrar's office directly.", "very_low"
    
    best = results[0]
    
    # Response based on confidence
    if best["confidence"] == "very_low":
        response = f"I'm not entirely sure, but this might be related:\n\n{best['answer']}\n\nPlease verify with the AAU Registrar's office for accuracy."
    elif best["confidence"] == "low":
        response = f"Based on my knowledge, here's what I found:\n\n{best['answer']}\n\nConfidence: Medium-Low. Please confirm with official sources."
    elif best["confidence"] == "medium":
        response = f"{best['answer']}\n\n---\n*I'm moderately confident about this answer.*"
    else:  # high confidence
        response = best["answer"]
    
    return response, best["confidence"]

# -----------------------------
# 6. UI Components
# -----------------------------
def setup_sidebar():
    """Create sidebar with information and controls"""
    with st.sidebar:
        st.image("https://www.aau.edu.et/themes/custom/aau/logo.png", width=200, use_column_width=True)
        st.markdown("## ℹ️ About")
        st.markdown("""
        This AI assistant uses:
        - **Sentence-BERT** for semantic understanding
        - **FAISS** for fast similarity search
        - **Streamlit** for chat interface
        
        Ask questions about:
        - 📝 Admissions
        - 📚 Registration  
        - 🗓️ Academic Calendar
        - 💰 Fees & Payments
        - 🎓 Student Services
        """)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Model: {MODEL_NAME}")
        st.caption("Version: 1.0.0")

def display_example_questions():
    """Show example questions in sidebar"""
    with st.sidebar:
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "When is the admission deadline?",
            "How do I register for courses?",
            "What is the academic calendar?",
            "How to get transcript?",
            "What are GPA requirements?",
            "Where is the main campus?"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q}"):
                return q
    return None

def streaming_response(text: str, delay: float = 0.008):
    """Display text with a typing effect"""
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(delay)
    placeholder.markdown(displayed_text)
    return displayed_text

# -----------------------------
# 7. Main Application
# -----------------------------
def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AAU AI Assistant - Semantic FAQ System",
        page_icon="🎓",
        layout="wide"
    )
    
    # Title
    st.title("🎓 AAU AI University Assistant")
    st.markdown("*Semantic FAQ System powered by Sentence-BERT and FAISS*")
    st.divider()
    
    # Load resources
    with st.spinner("Initializing the AI Assistant..."):
        model = load_model()
        questions, answers = load_faq_data()
        embeddings = build_embeddings(model, questions)
        index = build_faiss_index(embeddings)
    
    # Setup sidebar
    setup_sidebar()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Example questions
    example_query = display_example_questions()
    
    # Chat input
    user_query = st.chat_input("Ask me anything about Addis Ababa University...")
    
    # Use example query if clicked
    if example_query and not user_query:
        user_query = example_query
    
    # Process user query
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Search for answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching the knowledge base..."):
                results = search(user_query, model, index, questions, answers, top_k=3)
            
            # Format response
            response, confidence = format_response(results)
            
            # Display with streaming effect
            full_response = streaming_response(response)
            
            # Show alternative answers
            if len(results) > 1:
                with st.expander("🔍 See related questions"):
                    for i, result in enumerate(results[1:], 1):
                        st.markdown(f"**Option {i}:**")
                        st.markdown(f"*Q: {result['question']}*")
                        st.markdown(f"**A:** {result['answer']}")
                        st.caption(f"Similarity score: {result['score']:.2f} ({result['confidence']})")
                        if i < len(results[1:]):
                            st.divider()
            
            # Show confidence indicator
            if confidence == "high":
                st.success("✅ High confidence answer")
            elif confidence == "medium":
                st.info("📘 Medium confidence - please verify with official sources")
            elif confidence == "low":
                st.warning("⚠️ Low confidence - please confirm with AAU Registrar")
            else:
                st.error("❓ Very low confidence - consider rephrasing or contacting AAU directly")
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# -----------------------------
# 8. Run the app
# -----------------------------
if __name__ == "__main__":
    main()