import os
import streamlit as st
import pandas as pd
import base64
import torch

# -----------------------------------------------------------------------------
# Enable full CPU‐thread parallelism and GPU convolution tuning
# -----------------------------------------------------------------------------
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.benchmark = True

import numpy as np
import psycopg2
from io import BytesIO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from scoring_model import score_with_explanation
from nicknames import load_nickname_map_from_db, normalize_name
from vector_search import find_similar_faces

# Load model and processor
clip_model = CLIPModel.from_pretrained( "openai/clip-vit-base-patch32" )

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast_init=True
)
clip_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def encode_clip_vector(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = clip_model.get_image_features(**inputs)[0]
        vec = vec / vec.norm()
    return vec.cpu().numpy().tolist()

def log_user_feedback(input_record, match_id, score, label):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback_log (
            id SERIAL PRIMARY KEY,
            input_first TEXT, input_last TEXT, input_dob DATE, input_img TEXT,
            matched_id INT, match_score FLOAT, label INT
        );
    """)
    cur.execute("""
        INSERT INTO user_feedback_log (
            input_first, input_last, input_dob, input_img,
            matched_id, match_score, label
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        input_record["first_nm"], input_record["last_nm"], input_record["birth_dt"], input_record["headshot_b64"],
        match_id, score, label
    ))
    conn.commit()
    cur.close()
    conn.close()

# --- Streamlit UI ---
st.set_page_config(layout="centered")
st.title("🧬 Real-Time Deduplication with Vector Search")

with st.form("dedupe_form"):
    first_nm = st.text_input("First Name")
    last_nm = st.text_input("Last Name")
    birth_dt = st.date_input("Date of Birth")
    mdm_id = st.text_input("MDM Person ID")
    email = st.text_input("Email")
    headshot = st.file_uploader("Upload Headshot", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("🔍 Check for Duplicates")

if submit and headshot:
    st.info("🧠 Processing...")

    nickname_map = load_nickname_map_from_db()
    norm_first = normalize_name(first_nm, nickname_map)

    image_vec = encode_clip_vector(headshot)
    b64_img = base64.b64encode(headshot.getvalue()).decode("utf-8")

    # candidates_df = find_top_candidates(image_vec, k=25)
    candidates_df = find_similar_faces(image_vec, top_k=25)


    # Prepare for scoring
    pairs = []
    for _, row in candidates_df.iterrows():
        pairs.append({
            'a_id': 9999,
            'b_id': row['person_id'],
            'a_first': norm_first,
            'b_first': row['first_nm'],
            'a_last': last_nm,
            'b_last': row['last_nm'],
            'a_birth': birth_dt.strftime('%Y-%m-%d'),
            'b_birth': row['birth_dt'],
            'a_email': email,
            'b_email': row['email_address'],
            'a_mdm': mdm_id,
            'b_mdm': row['mdm_person_id'],
            'a_img': b64_img,
            'b_img': row['headshot_b64']
        })

    batch_df = pd.DataFrame(pairs)
    scored = score_with_explanation(batch_df)
    best_match = scored.sort_values("score", ascending=False).iloc[0]

    # Show match
    st.subheader(f"Top Match: {best_match['b_first']} {best_match['b_last']} — Score: {best_match['score']:.3f}")
    col1, col2 = st.columns(2)
    col1.image(base64.b64decode(best_match['a_img']), caption="Your Input", width=128)
    col2.image(base64.b64decode(best_match['b_img']), caption="Top Match", width=128)

    st.markdown("#### 🧠 Feature Contributions")
    feat_cols = [c for c in best_match.index if c.startswith("feat_")]
    st.dataframe(best_match[feat_cols].to_frame().T.rename(columns=lambda x: x.replace("feat_", "")))

    col3, col4 = st.columns(2)
    if col3.button("✅ Confirm Match"):
        log_user_feedback({
            "first_nm": norm_first,
            "last_nm": last_nm,
            "birth_dt": birth_dt,
            "headshot_b64": b64_img
        }, best_match["b_id"], best_match["score"], 1)
        st.success("✅ Match confirmed and logged")
    if col4.button("❌ Not a Match"):
        log_user_feedback({
            "first_nm": norm_first,
            "last_nm": last_nm,
            "birth_dt": birth_dt,
            "headshot_b64": b64_img
        }, best_match["b_id"], best_match["score"], 0)
        st.success("📝 Non-match logged")
