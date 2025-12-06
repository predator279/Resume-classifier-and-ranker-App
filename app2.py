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

# Note: The original TFIDFVectorizer import is removed as we use BERT embeddings now.
# import PyPDF2, docx are already imported.

# === Load Model, Tokenizer, and Label Encoder for Category Prediction ===
from huggingface_hub import hf_hub_download
@st.cache_resource
def load_model():
    # --- The REPOSITORY ID from Hugging Face ---
    MODEL_REPO_ID = "predator279/resume-classifier-model" # <-- **CONFIRM THIS IS YOUR CORRECT REPO ID*
    # 1. Load Model/Tokenizer from the Hub
    model = BertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_REPO_ID)
    
    # 2. Download and load the custom label encoder (.pkl) from the Hub
    label_encoder_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="label_encoder.pkl"
    )
    
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
        
    return model, tokenizer, le

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

# === BERT Embedding Function (New) ===
# === BERT Embedding Function (Updated with hash_funcs) ===
@st.cache_data(show_spinner=False, hash_funcs={
    BertForSequenceClassification: lambda _: None, # Ignore model entirely
    BertTokenizerFast: lambda _: None, # Ignore tokenizer entirely
    torch.device: lambda device: str(device) # Safely hash the device
})
def get_embedding(text, model, tokenizer, device):
    """Generates a fixed-size semantic embedding for a given text using BERT [CLS] token."""
    # ... (function body remains the same)
    cleaned = clean_text(text)
    if not cleaned:
        return np.zeros(768) 
        
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True) 
        cls_embedding = outputs.hidden_states[-1][:, 0, :] 
        
    return cls_embedding.cpu().numpy().flatten()

# === Skill Coverage Helper Function (New) ===
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


# === Resume Category Prediction (No Change) ===
def predict_category(text, model, tokenizer, le, device, top_k=5):
    # ... (function body remains the same)
    cleaned = clean_text(text)
    max_length = 512
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = le.inverse_transform(top_indices)
    top_scores = probs[top_indices]
    
    return list(zip(top_labels, top_scores))


# === Resume Ranking Based on Structured JD and BERT Embeddings (Replaced) ===
def rank_resumes_bert(job_structure, job_description_full, uploaded_files, model, tokenizer, device):
    
    # 1. Pre-calculate JD embeddings
    jd_emb = get_embedding(job_description_full, model, tokenizer, device)
    
    # 2. Pre-calculate requirement embeddings
    for req in job_structure["requirements"]:
        req['embedding'] = get_embedding(req["text"], model, tokenizer, device)
        
    resume_scores = []
    
    for file in uploaded_files:
        resume_raw = extract_text_from_file(file)
        resume_text = clean_text(resume_raw)
        
        # A. Get Resume Embedding
        resume_emb = get_embedding(resume_text, model, tokenizer, device)
        
        # B. Calculate Compliance Score (Weighted Requirements Match)
        total_weight = sum(req["weight"] for req in job_structure["requirements"])
        score_sum = 0.0
        hard_fail = False
        breakdown = {}

        for req in job_structure["requirements"]:
            # 1. Semantic Similarity (cosine similarity between requirement text and full resume)
            sim_raw = cosine_similarity(req['embedding'].reshape(1, -1), resume_emb.reshape(1, -1))[0][0]
            # Normalize [-1, 1] to [0, 1] for scoring
            sim = max(0.0, min((sim_raw + 1) / 2, 1.0)) 

            # 2. Skill Coverage (only for skill type)
            coverage = 1.0
            if req["type"] == "skill":
                coverage = get_skill_coverage(req["text"], resume_text)

            # 3. Combine: 70% Semantic Match + 30% Keyword Coverage (tuneable)
            req_score = 0.7 * sim + 0.3 * coverage
            
            # Hard Rule Check: If must-have score is low
            if req.get("must_have", False) and req_score < 0.4: 
                hard_fail = True
            
            score_sum += req["weight"] * req_score
            breakdown[req["type"]] = {"score": req_score, "text": req["text"]}

        comp_score = score_sum / max(1e-6, total_weight)
        
        # Apply Hard Filter Cap
        if hard_fail:
            comp_score = min(comp_score, 0.45) # Cap at 45% if must-haves are critically missing

        # C. Global Semantic Similarity
        global_sim_raw = cosine_similarity(jd_emb.reshape(1, -1), resume_emb.reshape(1, -1))[0][0]
        global_sim = max(0.0, min((global_sim_raw + 1) / 2, 1.0))

        # D. Final Score (0.4 Global Match + 0.6 Compliance)
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
st.title("🔍 AI-Powered Resume Analyzer")

# Sidebar for navigation
mode = st.sidebar.radio("Choose Mode:", ["Resume Category Prediction", "Resume Ranking"])

# Category Prediction Mode
if mode == "Resume Category Prediction":
    st.subheader("Resume Category Prediction (BERT)")

    uploaded_file = st.file_uploader("Upload your resume (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"])

    if uploaded_file is not None:
        model, tokenizer, le = load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # CONSOLIDATED FILE EXTRACTION
        resume_text = extract_text_from_file(uploaded_file)

        if resume_text:
            st.subheader("📄 Resume Preview")
            st.write(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)

            st.subheader("🔮 Top 5 Predicted Categories")
            top5 = predict_category(resume_text, model, tokenizer, le, device)
            for i, (label, score) in enumerate(top5, 1):
                st.write(f"**{i}. {label}** — Score: {score:.4f}")
        else:
            st.error("❌ Unsupported file type or empty content.")

# Ranking Mode - UPDATED
elif mode == "Resume Ranking":
    st.subheader("Resume Ranking (Hybrid BERT Score)")
    
    st.markdown("##### 📝 Structured Job Requirements")
    
    col1, col2 = st.columns(2)
    with col1:
        job_description_skills = st.text_area("Required Skills (Must-Have, e.g., Python, SQL, AWS)", height=100)
    with col2:
        job_description_experience = st.text_area("Min. Experience/Keywords (e.g., 2+ years data science)", height=100)
    
    job_description_education = st.text_input("Education (e.g., B.Tech CS or related)", value="Relevant degree required")
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
    job_description_full = f"Skills: {job_description_skills}. Experience: {job_description_experience}. Education: {job_description_education}. Other: {job_description_other}"


    if st.button("🚀 Rank Resumes") and job_description_full and uploaded_resumes:
        # Load model and set device
        model, tokenizer, le = load_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Call the new BERT-based ranker
        with st.spinner("Processing Resumes..."):
            ranked_results = rank_resumes_bert(job_structure, job_description_full, uploaded_resumes, model, tokenizer, device)

        st.subheader("📊 Ranked Resumes:")
        
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
                    
                    # Displaying the first 70 characters of the requirement text
                    req_display = req_data['text'][:70].replace('\n', ' ') + "..."
                    
                    st.markdown(f"{icon} **{req_type.capitalize()}** (`{req_display}`): **{req_data['score']:.4f}**")
