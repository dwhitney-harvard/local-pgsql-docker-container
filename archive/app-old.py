import streamlit as st
import pandas as pd
import base64
from archive.scoring import score_matches
from nicknames import load_nickname_map

st.set_page_config(layout="wide")
st.title("👥 Duplicate Detection Review")

score_threshold = st.slider("Minimum score to show", 0.0, 1.0, 0.85, step=0.01)

uploaded_file = st.file_uploader("📤 Upload a batch CSV file", type=["csv"])

if uploaded_file:
    batch = pd.read_csv(uploaded_file)
    golden = pd.read_pickle("people_with_embeddings.pkl")

    if "embedding" not in batch.columns:
        st.error("Batch CSV must contain precomputed embeddings.")
    else:
        results = score_matches(batch, golden)
        filtered = [r for r in results if r["score"] >= score_threshold]

        st.success(f"✅ Showing {len(filtered)} matches above threshold.")

        for i, r in enumerate(filtered):
            st.subheader(f"{r['input_name']} ⇄ {r['match_name']}")
            st.markdown(f"**Score:** `{r['score']}`")

            col1, col2 = st.columns(2)
            col1.image(base64.b64decode(r["input_img"]), width=128, caption="Input")
            col2.image(base64.b64decode(r["match_img"]), width=128, caption="Match")

            col3, col4 = st.columns(2)
            if col3.button("✅ Confirm Match", key=f"yes-{i}"):
                with open("user_feedback_log.csv", "a") as f:
                    f.write(f"{r['input_person_id']},{r['match_person_id']},1\n")
            if col4.button("❌ Reject Match", key=f"no-{i}"):
                with open("user_feedback_log.csv", "a") as f:
                    f.write(f"{r['input_person_id']},{r['match_person_id']},0\n")

            st.divider()
