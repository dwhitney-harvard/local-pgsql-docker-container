import pandas as pd
import joblib
from extract_features import extract_features

# Load model once
model = joblib.load("dedup_model.pkl")

def score_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    X, _ = extract_features(batch_df)
    batch_df['score'] = model.predict_proba(X)[:, 1]
    return batch_df.sort_values(by="score", ascending=False)

def score_with_explanation(batch_df: pd.DataFrame) -> pd.DataFrame:
    X, _ = extract_features(batch_df)
    probs = model.predict_proba(X)[:, 1]
    batch_df['score'] = probs
    for i, col in enumerate(X.columns):
        batch_df[f"feat_{col}"] = X[col]
    return batch_df