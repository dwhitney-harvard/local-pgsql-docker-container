from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.fuzz import partial_ratio
import numpy as np
from nicknames import load_nickname_map, normalize_name

nickname_map = load_nickname_map("data/converted_names.csv")

def text_similarity(a, b):
    return partial_ratio(a, b) / 100.0

def score_matches(batch_df, db_df, top_k=3):
    matches = []
    db_embeddings = np.stack(db_df['embedding'].to_numpy())

    for i, row in batch_df.iterrows():
        input_embedding = np.array(row['embedding']).reshape(1, -1)
        sims = cosine_similarity(input_embedding, db_embeddings)[0]
        top_idxs = sims.argsort()[-top_k:][::-1]

        for idx in top_idxs:
            db_row = db_df.iloc[idx]
            sim = sims[idx]

            norm_input = normalize_name(row['first_nm'], nickname_map)
            norm_db = normalize_name(db_row['first_nm'], nickname_map)
            name_sim = text_similarity(norm_input, norm_db)

            final_score = 0.7 * sim + 0.3 * name_sim

            matches.append({
                "input_person_id": row['person_id'],
                "match_person_id": db_row['person_id'],
                "input_name": f"{row['first_nm']} {row['last_nm']}",
                "match_name": f"{db_row['first_nm']} {db_row['last_nm']}",
                "input_img": row['headshot_b64'],
                "match_img": db_row['headshot_b64'],
                "score": round(final_score, 4)
            })

    return matches
