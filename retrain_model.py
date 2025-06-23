import pandas as pd
import psycopg2
import joblib
from train_model import train_model_from_df
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

MODEL_OUTPUT = "dedup_model.pkl"

# Extract feedback-labeled pairs and rebuild feature vectors
def fetch_feedback_pairs():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    cur.execute("""
        SELECT input_first, input_last, input_dob, input_img,
               matched_id, match_score, label
        FROM user_feedback_log
    """)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns)

def rebuild_training_df(feedback_df):
    import base64
    from extract_features import extract_feature_vector_for_pair
    from vector_search import get_candidate_by_id

    pairs = []
    for _, row in feedback_df.iterrows():
        b_id = row["matched_id"]
        b_record = get_candidate_by_id(b_id)
        if b_record is None:
            continue

        pair = {
            'a_first': row['input_first'],
            'b_first': b_record['first_nm'],
            'a_last': row['input_last'],
            'b_last': b_record['last_nm'],
            'a_birth': row['input_dob'],
            'b_birth': b_record['birth_dt'],
            'a_email': None,
            'b_email': b_record['email_address'],
            'a_mdm': None,
            'b_mdm': b_record['mdm_person_id'],
            'a_img': row['input_img'],
            'b_img': b_record['headshot_b64'],
            'label': row['label']
        }
        vec = extract_feature_vector_for_pair(pair)
        vec['label'] = row['label']
        pairs.append(vec)

    return pd.DataFrame(pairs)

if __name__ == "__main__":
    feedback_df = fetch_feedback_pairs()
    print(f"📊 Found {len(feedback_df)} labeled feedback pairs")

    if len(feedback_df) < 10:
        print("⚠️ Not enough feedback samples to retrain. Skipping.")
        exit()

    training_df = rebuild_training_df(feedback_df)
    print(f"✅ Rebuilt training set with {len(training_df)} samples")

    # Split data into training and test sets
    X = training_df.drop(columns=["label"])
    y = training_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrain model
    model = train_model_from_df(pd.concat([X_train, y_train], axis=1))
    joblib.dump(model, MODEL_OUTPUT)
    print(f"✅ Saved retrained model to {MODEL_OUTPUT}")

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n📈 Validation Report:")
    print(classification_report(y_test, y_pred))
    print("✅ Model retraining complete")