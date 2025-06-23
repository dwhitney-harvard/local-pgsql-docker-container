import psycopg2

def load_nickname_map_from_db():
    DB = {
        "host": "localhost",
        "port": 5432,
        "dbname": "mydb",
        "user": "postgres",
        "password": "mypassword"
    }

    nickname_dict = {}
    try:
        conn = psycopg2.connect(**DB)
        cur = conn.cursor()
        cur.execute("SELECT nickname, canonical FROM nicknames")
        for nickname, canonical in cur.fetchall():
            nickname_dict[nickname.lower()] = canonical.lower()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error loading nickname map from DB: {e}")

    return nickname_dict

def normalize_name(name, nickname_map):
    return nickname_map.get(name.lower(), name.lower())
