import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from PIL import Image
from io import BytesIO
from nicknames import load_nickname_map_from_db, normalize_name
from hybrid_search import rerank_with_text
from vector_search import find_similar_faces
from embedding_cache import encode_image_b64_to_vector
from scoring_model import score_with_explanation

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def log_user_feedback(input_record, match_id, score, label):
    import psycopg2
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

st.set_page_config(layout="centered")
st.title("🔍 Hybrid Duplicate Finder")

with st.form("dedupe_form"):
    first_nm = st.text_input("First Name")
    last_nm = st.text_input("Last Name")
    birth_dt = st.date_input("Date of Birth")
    mdm_id = st.text_input("MDM Person ID")
    email = st.text_input("Email")
    headshot = st.file_uploader("Upload Headshot", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Check for Matches")

if submit and headshot:
    st.info("Processing search and scoring...")
    b64_img = base64.b64encode(headshot.getvalue()).decode("utf-8")
    img_vec = encode_image_b64_to_vector(b64_img)

    norm_first = normalize_name(first_nm, load_nickname_map_from_db())
    
    candidates = find_similar_faces(img_vec, top_k=10)
    
    reranked = rerank_with_text({
        "first_nm": norm_first,
        "last_nm": last_nm,
        "mdm_person_id": mdm_id,
        "email_address": email
    }, candidates)

    pairs = []
    for _, row in reranked.iterrows():
        pairs.append({
            'a_id': 0,
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

    scored = score_with_explanation(pd.DataFrame(pairs))
    top = scored.sort_values("score", ascending=False).iloc[0]

    st.subheader(f"Top Match: {top['b_first']} {top['b_last']} — Score: {top['score']:.3f}")
    col1, col2 = st.columns(2)
    col1.image(base64.b64decode(top['a_img']), caption="Your Input", width=128)
    
    col2.image(base64.b64decode(top['b_img']), width=128)
    col2.markdown(f"**Name:** {top['b_first']} {top['b_last']}\n"
                  f"**MDM ID:** {top['b_mdm']}\n"
                  f"**Email:** {top['b_email']}\n"
                  f"**Birth Date:** {top['b_birth']}\n")

    st.markdown("#### Feature Contributions")
    st.dataframe(top[[c for c in top.index if c.startswith("feat_")]].to_frame().T.rename(columns=lambda x: x.replace("feat_", "")))

    if st.button("✅ Confirm Match"):
        log_user_feedback({
            "first_nm": norm_first,
            "last_nm": last_nm,
            "birth_dt": birth_dt,
            "headshot_b64": b64_img
        }, top['b_id'], top['score'], 1)
        st.success("Match confirmed")

    if st.button("❌ Not a Match"):
        log_user_feedback({
            "first_nm": norm_first,
            "last_nm": last_nm,
            "birth_dt": birth_dt,
            "headshot_b64": b64_img
        }, top['b_id'], top['score'], 0)
        st.success("Feedback recorded")
