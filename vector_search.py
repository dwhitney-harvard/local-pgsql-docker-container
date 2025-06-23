# vector_search.py

import psycopg2
import pandas as pd
from psycopg2 import sql

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def connect_db():
    return psycopg2.connect(**DB)

def find_similar_faces(vec, top_k=10):
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    vec_literal = "[" + ",".join(map(str, vec)) + "]"

    cur.execute("""
        SELECT
          person_id,
          first_nm,
          last_nm,
          birth_dt,
          mdm_person_id,
          email_address,
          headshot_b64,
          face_embedding <-> %s::vector AS distance
        FROM people_with_faces
        ORDER BY face_embedding <-> %s::vector
        LIMIT %s
    """, (vec_literal, vec_literal, top_k))

    rows = cur.fetchall()
    
    # pull column names dynamically
    col_names = [desc[0] for desc in cur.description]
    
    cur.close()
    conn.close()
    return pd.DataFrame(
        rows,
        columns=col_names
        # columns=[
        #   "person_id",
        #   "first_nm",
        #   "last_nm",
        #   "birth_dt",
        #   "mdm_person_id",
        #   "email_address",
        #   "headshot_b64",
        #   "distance"
        # ]
    )


def find_similar_textual(first_nm=None, last_nm=None, email=None, mdm=None, dob=None, top_k=50):
    filters = []
    params = []

    if first_nm:
        filters.append("first_nm ILIKE %s")
        params.append(f"{first_nm}%")

    if last_nm:
        filters.append("last_nm ILIKE %s")
        params.append(f"{last_nm}%")

    if email:
        filters.append("email_address ILIKE %s")
        params.append(f"{email}%")

    if mdm:
        filters.append("CAST(mdm_person_id AS TEXT) ILIKE %s")
        params.append(f"{mdm}%")

    if dob:
        filters.append("birth_dt = %s")
        params.append(dob)

    if not filters:
        return pd.DataFrame()  # Avoid full table scan

    query = sql.SQL("""SELECT * FROM people_with_faces WHERE {conditions} LIMIT %s""").format( conditions=sql.SQL(" OR ").join(map(sql.SQL, filters)))

    params.append(top_k)

    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        cur.close()
        conn.close()