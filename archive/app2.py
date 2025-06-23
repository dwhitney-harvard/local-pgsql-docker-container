import streamlit as st
import pandas as pd
import base64
import psycopg2
from io import BytesIO
from PIL import Image
# from scoring_model import score_batch
from scoring_model import score_with_explanation
from nicknames import load_nickname_map_from_db, normalize_name

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def fetch_golden_people():
    conn = psycopg2.connect(**DB)
    df = pd.read_sql("SELECT * FROM people_with_faces", conn)
    conn.close()
    return df

def log_user_submission(record, match_id, score, label):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback_log (
            id SERIAL PRIMARY KEY,
            input_first TEXT, input_last TEXT, input_dob DATE, input_img TEXT,
            matched_id INT, match_score FLOAT, label INT
        )
    """)
    cur.execute("""
        INSERT INTO user_feedback_log (
            input_first, input_last, input_dob, input_img,
            matched_id, match_score, label
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        record["first_nm"], record["last_nm"], record["birth_dt"], record["headshot_b64"],
        match_id, score, label
    ))
    conn.commit()
    cur.close()
    conn.close()

def encode_image_to_b64(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# UI
st.set_page_config(layout="centered")
st.title("🧬 Real-Time Duplicate Checker")

with st.form("match_form"):
    first = st.text_input("First Name")
    last = st.text_input("Last Name")
    dob = st.date_input("Birthdate")
    email = st.text_input("Email")
    mdm = st.text_input("MDM Person ID")
    headshot = st.file_uploader("Upload Headshot Image", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("🔍 Find Matches")

if submitted and headshot:
    nickname_map = load_nickname_map_from_db()
    norm_first = normalize_name(first, nickname_map)
    headshot_b64 = encode_image_to_b64(headshot)

    golden_df = fetch_golden_people()

    # Create pairwise inputs for scoring
    batch_df = pd.DataFrame([{
        'a_id': 9999,  # placeholder
        'b_id': row['person_id'],
        'a_first': norm_first,
        'b_first': row['first_nm'],
        'a_last': last,
        'b_last': row['last_nm'],
        'a_birth': dob.strftime('%Y-%m-%d'),
        'b_birth': row['birth_dt'],
        'a_email': email,
        'b_email': row['email_address'],
        'a_mdm': mdm,
        'b_mdm': row['mdm_person_id'],
        'a_img': headshot_b64,
        'b_img': row['headshot_b64']
    } for _, row in golden_df.iterrows()])

    # scored = score_batch(batch_df)
    scored = score_with_explanation(batch_df)

    best_match = scored.iloc[0]

    st.markdown(f"### Best Match (Score: `{best_match['score']:.3f}`)")
    col1, col2 = st.columns(2)
    col1.image(base64.b64decode(best_match['a_img']), caption="Your Input", width=128)
    col2.image(base64.b64decode(best_match['b_img']), caption="Top Match", width=128)

    st.write(f"🧍 Match: {best_match['b_first']} {best_match['b_last']} | DOB: {best_match['b_birth']}")

    col3, col4 = st.columns(2)
    if col3.button("✅ Confirm Match"):
        log_user_submission({
            "first_nm": norm_first, "last_nm": last, "birth_dt": dob,
            "headshot_b64": headshot_b64
        }, best_match["b_id"], best_match["score"], 1)
        st.success("✅ Match confirmed and logged.")
    if col4.button("❌ Not a Match"):
        log_user_submission({
            "first_nm": norm_first, "last_nm": last, "birth_dt": dob,
            "headshot_b64": headshot_b64
        }, best_match["b_id"], best_match["score"], 0)
        st.info("📝 Feedback recorded as non-match.")
