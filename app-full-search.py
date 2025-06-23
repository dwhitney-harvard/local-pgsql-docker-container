import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from nicknames import load_nickname_map_from_db, normalize_name
from hybrid_search import rerank_with_text
from vector_search import find_similar_faces, find_similar_textual
from embedding_cache import encode_image_b64_to_vector
from scoring_model import score_with_explanation

nickname_map = load_nickname_map_from_db()

st.set_page_config(layout="wide")
st.title("🔍 Hybrid Duplicate Finder")

with st.form("dedupe_form"):
    first_nm = st.text_input("First Name")
    last_nm = st.text_input("Last Name")
    birth_dt = st.date_input("Date of Birth")
    email = st.text_input("Email")
    mdm_id = st.text_input("MDM Person ID")
    headshot = st.file_uploader("Upload Headshot (optional)", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Check for Matches")

if submit:
    st.info("🔎 Searching for potential duplicates...")

    norm_first = normalize_name(first_nm, nickname_map)

    # Start building query fields
    input_record = {
        "first_nm": norm_first,
        "last_nm": last_nm,
        "birth_dt": birth_dt.strftime("%Y-%m-%d"),
        "email_address": email,
        "mdm_person_id": mdm_id,
        "headshot_b64": None,
        "embedding": None
    }

    candidates = pd.DataFrame()

    # Image-based candidate retrieval
    if headshot:
        b64_img = base64.b64encode(headshot.getvalue()).decode("utf-8")
        img_vec = encode_image_b64_to_vector(b64_img)
        input_record["headshot_b64"] = b64_img
        input_record["embedding"] = img_vec
        candidates = find_similar_faces(img_vec, top_k=50)

    # Textual-based fallback or merge
    text_candidates = find_similar_textual(norm_first, last_nm, email, mdm_id, birth_dt.strftime("%Y-%m-%d"), top_k=50)
    candidates = pd.concat([candidates, text_candidates]).drop_duplicates("person_id")

    # Rerank with trained model
    reranked = rerank_with_text(input_record, candidates)

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
            'a_img': input_record['headshot_b64'],
            'b_img': row['headshot_b64']
        })

    scored = score_with_explanation(pd.DataFrame(pairs))
    top_n = scored.sort_values("score", ascending=False).head(5)

    st.subheader(f"🧠 Top {len(top_n)} Matches")
    for i, row in top_n.iterrows():
        st.markdown(f"### 🎯 Match #{i+1} — Score: {row['score']:.3f}")
        col1, col2 = st.columns(2)
        col1.markdown(f"**Input**: {row['a_first']} {row['a_last']} ({row['a_birth']})")
        if row['a_img']:
            col1.image(base64.b64decode(row['a_img']), width=128)
        col2.markdown(f"**Match**: {row['b_first']} {row['b_last']} ({row['b_birth']})")
        if row['b_img']:
            col2.image(base64.b64decode(row['b_img']), width=128)

        st.markdown("**Feature Contributions:**")
        features = row[[c for c in row.index if c.startswith("feat_")]]
        st.dataframe(features.to_frame().T.rename(columns=lambda x: x.replace("feat_", "")), use_container_width=True)

        st.markdown("---")
