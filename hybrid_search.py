# hybrid_search.py

import pandas as pd
from vector_search import find_similar_faces
# from extract_features import cosine_sim, jaccard_sim
from difflib import SequenceMatcher

def text_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def rerank_with_text(user_record, candidates_df):
    def score_row(row):
        score = 0
        if user_record.get("first_nm") and row["first_nm"]:
            score += text_similarity(user_record["first_nm"], row["first_nm"]) * 0.25
        if user_record.get("last_nm") and row["last_nm"]:
            score += text_similarity(user_record["last_nm"], row["last_nm"]) * 0.25
        if user_record.get("email_address") and row["email_address"]:
            score += text_similarity(user_record["email_address"], row["email_address"]) * 0.2
        if user_record.get("mdm_person_id") and row["mdm_person_id"]:
            score += int(user_record["mdm_person_id"] == row["mdm_person_id"]) * 0.3
        return score

    candidates_df["text_score"] = candidates_df.apply(score_row, axis=1)
    return candidates_df.sort_values("text_score", ascending=False)
