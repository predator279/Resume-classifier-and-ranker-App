import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document
from io import StringIO

# 1. NEW IMPORT: For high-quality semantic ranking
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# === Load Model, Tokenizer, and Label Encoder for Category Prediction & Ranking ===
@st.cache_resource
def load_model():
    # --- The REPOSITORY ID from Hugging Face ---
    MODEL_REPO_ID = "predator279/resume-classifier-model" # <-- CONFIRM THIS IS YOUR CORRECT REPO ID
    
    # 1. Load Classification Model and Tokenizer from the Hub (FOR CATEGORY PREDICTION)
    model_cls = BertForSequenceClassification.from_pretrained(MODEL_REPO_ID) # <-- RENAMED TO model_cls
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_REPO_ID)
    
    # 2. Download and load the custom label encoder (.pkl) from the Hub
    label_encoder_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="label_encoder.pkl"
    )
    
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)

    # 3. NEW: Load the Sentence Transformer model for Semantic Ranking
    RANKING_MODEL_NAME = 'all-MiniLM-L6-v2' # Excellent balance of size/speed/performance
    model_rank = SentenceTransformer(RANKING_MODEL_NAME)
    
    # 4. UPDATED RETURN: Return all four necessary components
    return model_cls, tokenizer, le, model_rank # <-- CRITICAL CHANGE HERE

# === Text Cleaning Function for Both Categories and Ranking ===
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'RT|cc', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === Consolidated Function to Extract Text from Files (used for both modes) ===
def extract_text_from_file(file):
    # Rewritten to use standard libraries imported
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return " ".join([page.extract_text() or '' for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        # Streamlit file uploader returns BytesIO, need to decode
        return file.getvalue().decode("utf-8")
    else:
        return ""

# === BERT Embedding Function (UPDATED: Uses Sentence Transformer Model) ===
@st.cache_data(show_spinner=False, hash_funcs={
    # Only need to ignore the complex SentenceTransformer object for caching
    SentenceTransformer: lambda _: None, 
    torch.device: lambda device: str(device) 
})
def get_embedding(text, model_rank, device): # <-- UPDATED ARGUMENTS!
    """Generates a fixed-size semantic embedding for a given text using SentenceTransformer."""
    cleaned = clean_text(text)
    if not cleaned:
        # MiniLM-L6-v2 produces a 384-dimensional vector
        return np.zeros(384) 
    
    # Move model to device, encode, and move back
    model_rank.to(device)
    embeddings = model_rank.encode([cleaned], convert_to_numpy=True, show_progress_bar=False)
    model_rank.to('cpu') 
    
    return embeddings[0].flatten()

# === Skill Coverage Helper Function (No Change) ===
def get_skill_coverage(req_text, resume_text):
    """Calculates keyword presence as a fractional score (simple approach)."""
    # Clean and convert comma-separated required skills to a set of lower-cased tokens
    req_skills = set(s.strip().lower() for s in req_text.split(',') if s.strip())
    if not req_skills:
        return 1.0 # No skills required
        
    resume_text_lower = clean_text(resume_text).lower()
    
    present_count = 0
    for skill in req_skills:
        # Check for exact skill/phrase presence in the cleaned resume text
        if skill in resume_text_lower:
             present_count += 1
             
    return present_count / len(req_skills)


# === Resume Category Prediction (No Change in logic, but change in args) ===
def predict_category(text, model_cls, tokenizer, le, device, top_k=5): # <-- UPDATED ARG NAME
    cleaned = clean_text(text)
    max_length = 512
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model_cls(**inputs).logits # <-- USE model_cls
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = le.inverse_transform(top_indices)
    top_scores = probs[top_indices]
    
    return list(zip(top_labels, top_scores))

# === Resume Ranking Based on Structured JD and BERT Embeddings (UPDATED: Uses model_rank) ===
def rank_resumes_bert(job_structure, job_description_full, uploaded_files, model_rank, tokenizer, device): # <-- UPDATED ARG NAME
    
    # 1. Filter out requirements with empty text
    valid_requirements = []
    for req in job_structure["requirements"]:
        if req["text"].strip(): # Only include if text is not empty
            req['embedding'] = get_embedding(req["text"], model_rank, device) # <-- PASS model_rank
            valid_requirements.append(req)
    
    # Pre-calculate JD embeddings for global match
    jd_emb = get_embedding(job_description_full, model_rank, device) # <-- PASS model_rank
    
    resume_scores = []
    
    for file in uploaded_files:
        resume_raw = extract_text_from_file(file)
        resume_text = clean_text(resume_raw)
        
        # A. Get Resume Embedding
        resume_emb = get_embedding(resume_text, model_rank, device) # <-- PASS model_rank
        
        # C. Global Semantic Similarity (Calculated first, used as fallback)
        global_sim_raw = cosine_similarity(jd_emb.reshape(1, -1), resume_emb.reshape(1, -1))[0][0]
        global_sim = max(0.0, min((global_sim_raw + 1) / 2, 1.0))
        
        # B. Calculate Compliance Score (Weighted Requirements Match)
        total_weight = sum(req["weight"] for req in valid_requirements)
        score_sum = 0.0
        hard_fail = False
        breakdown = {}
        
        # CRITICAL FIX: Handle case where ALL structured fields are empty
        if total_weight == 0:
            comp_score = global_sim
            final_score = global_sim
            
            breakdown['Global'] = {"score": global_sim, "text": "Score based only on Global JD Match"}
            
        else: # Normal calculation
            for req in valid_requirements:
                # 1. Semantic Similarity
                sim_raw = cosine_similarity(req['embedding'].reshape(1, -1), resume_emb.reshape(1, -1))[0][0]
                sim = max(0.0, min((sim_raw + 1) / 2, 1.0)) 

                # 2. Skill Coverage
                coverage = 1.0
                if req["type"] == "skill":
                    coverage = get_skill_coverage(req["text"], resume_text)

                # 3. Combine
                req_score = 0.7 * sim + 0.3 * coverage
                
                # Hard Rule Check
                if req.get("must_have", False) and req_score < 0.4: 
                    hard_fail = True
                
                score_sum += req["weight"] * req_score
                breakdown[req["type"]] = {"score": req_score, "text": req["text"]}

            comp_score = score_sum / total_weight
            
            # Apply Hard Filter Cap
            if hard_fail:
                comp_score = min(comp_score, 0.45) 
                
            # D. Final Score
            alpha = 0.4  
            beta  = 0.6  
            final_score = alpha * global_sim + beta * comp_score
        
        resume_scores.append({
            "name": file.name,
            "final_score": round(final_score * 100, 2), # 0-100 score
            "global_sim": global_sim,
            "compliance_score": comp_score,
            "breakdown": breakdown
        })

    # Rank and sort by final_score
    ranked_list = sorted(resume_scores, key=lambda x: x["final_score"], reverse=True)
    return ranked_list


# === Streamlit UI ===
st.set_page_config(page_title="Resume Analyzer", layout="centered")
st.title("🔍 AI-Powered Resume Analyzer v2")

# Sidebar for navigation
mode = st.sidebar.radio("Choose Mode:", ["Resume Category Prediction", "Resume Ranking"])

# Category Prediction Mode
if mode == "Resume Category Prediction":
    st.subheader("Resume Category Prediction (BERT)")

    uploaded_file = st.file_uploader("Upload your resume (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        # Load all four components
        model_cls, tokenizer, le, model_rank = load_model() # <-- UPDATED UNPACKING
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cls.to(device) # <-- USE model_cls
        model_cls.eval() # <-- USE model_cls

        # CONSOLIDATED FILE EXTRACTION
        resume_text = extract_text_from_file(uploaded_file)

        if resume_text:
            st.subheader("📄 Resume Preview")
            st.write(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

            st.subheader("🔮 Top 5 Predicted Categories")
            top5 = predict_category(resume_text, model_cls, tokenizer, le, device) # <-- USE model_cls
            for i, (label, score) in enumerate(top5, 1):
                st.write(f"**{i}. {label}** — Score: {score:.4f}")
        else:
            st.error("❌ Unsupported file type or empty content.")

# Ranking Mode - UPDATED
elif mode == "Resume Ranking":
    st.subheader("Resume Ranking (Hybrid BERT Score)")
    
    # --- Input Fields ---
    st.markdown("##### 📝 Structured Job Requirements (Must be filled to calculate Compliance Score)")
    
    col1, col2 = st.columns(2)
    with col1:
        job_description_skills = st.text_area("Required Skills (Must-Have, e.g., Python, SQL, AWS)", height=100)
    with col2:
        job_description_experience = st.text_area("Min. Experience/Keywords (e.g., 2+ years data science)", height=100)
    
    job_description_education = st.text_input("Education (e.g., B.Tech CS or related)", value="") 
    job_description_other = st.text_area("Other JD Text (for Global Matching)", height=100)
    
    uploaded_resumes = st.file_uploader("📄 Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # --- Construct Job Structure ---
    job_structure = {
      "requirements": [
        {"type": "skill", "text": job_description_skills, "weight": 0.40, "must_have": True},
        {"type": "experience", "text": job_description_experience, "weight": 0.35, "must_have": False},
        {"type": "education", "text": job_description_education, "weight": 0.25, "must_have": False},
      ]
    }
    
    job_description_full = f"{job_description_skills} {job_description_experience} {job_description_education} {job_description_other}"
    core_jd_text = job_description_full.strip()

    if st.button("🚀 Rank Resumes"):
        # NEW VALIDATION
        if not uploaded_resumes:
            st.error("❌ Please upload at least one resume.")
        elif not core_jd_text:
            st.error("❌ The Job Description is empty. Please enter requirements or general JD text.")
        else:
            # Load all four components
            model_cls, tokenizer, le, model_rank = load_model() # <-- UPDATED UNPACKING
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Note: SentenceTransformer model_rank will be moved to device inside get_embedding for a short time

            # Call the new BERT-based ranker
            with st.spinner("Processing Resumes..."):
                # PASS model_rank for semantic scoring
                ranked_results = rank_resumes_bert(job_structure, core_jd_text, uploaded_resumes, model_rank, tokenizer, device) 

            st.subheader("📊 Ranked Resumes:")
            
            # Display results (no changes here)
            for rank, result in enumerate(ranked_results):
                score_color = "green" if result['final_score'] > 75 else ("orange" if result['final_score'] > 50 else "red")
                
                st.markdown(
                    f"**{rank + 1}. {result['name']}** — **Total Match Score: <span style='color:{score_color}; font-size: 1.2em;'>{result['final_score']:.2f} / 100</span>**", 
                    unsafe_allow_html=True
                )
                
                with st.expander(f"🔍 Detailed Score Breakdown for {result['name']}"):
                    st.write(f"**Global JD Match (BERT Semantic Similarity):** {result['global_sim']:.4f}")
                    st.write(f"**Weighted Requirement Compliance:** {result['compliance_score']:.4f}")
                    st.markdown("---")
                    st.markdown("**Requirement-Specific Scores (Compliance Breakdown):**")
                    
                    for req_type, req_data in result['breakdown'].items():
                        icon = "✅" if req_data['score'] > 0.7 else ("⚠️" if req_data['score'] > 0.45 else "❌")
                        
                        req_display = req_data['text'][:70].replace('\n', ' ') + "..."
                        
                        st.markdown(f"{icon} **{req_type.capitalize()}** (`{req_display}`): **{req_data['score']:.4f}**")
