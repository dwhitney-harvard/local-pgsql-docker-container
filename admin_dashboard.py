import streamlit as st
import pandas as pd
import psycopg2
import joblib
import shutil
from datetime import datetime

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def load_logs():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, sample_count, f1_score, deployed
        FROM model_training_log
        ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns)

def promote_model(backup_name="dedup_model_backup.pkl"):
    shutil.copy("dedup_model.pkl", backup_name)
    st.success(f"✅ Model promoted manually and backed up to {backup_name}")

def override_deployment_flag():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        UPDATE model_training_log
        SET deployed = TRUE
        WHERE id = (
            SELECT id FROM model_training_log
            ORDER BY timestamp DESC
            LIMIT 1
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

st.set_page_config(layout="wide")
st.title("📊 Model Training Admin Dashboard")

logs_df = load_logs()

st.subheader("📈 Training History")
st.dataframe(logs_df, use_container_width=True)

st.markdown("---")
st.subheader("🔧 Manual Actions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Promote Current Model"):
        promote_model()

with col2:
    if st.button("Force Last Run as Deployed"):
        override_deployment_flag()
        st.success("Deployment flag updated in log table")
