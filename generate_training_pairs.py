import psycopg2
import pandas as pd
import random
from datetime import timedelta
from io import BytesIO
from PIL import Image, ImageOps
import base64

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def fetch_data():
    conn = psycopg2.connect(**DB)
    people = pd.read_sql("SELECT id, person_id, first_nm, last_nm, birth_dt, mdm_person_id, email_address, headshot_b64 FROM people_with_faces", conn)
    nick_df = pd.read_sql("SELECT nickname, canonical FROM nicknames", conn)
    conn.close()
    nickname_map = {n.lower(): c.lower() for n, c in nick_df.to_records(index=False)}
    return people, nickname_map

def flip_img_b64(b64):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img = ImageOps.mirror(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def rotate_img_b64(b64, angle=15):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img = img.rotate(angle, expand=True)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def rotate_180_b64(b64, angle=180):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img = img.rotate(angle, expand=True)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_training_data(df, nickname_map):
    pairs = []

    for _, row in df.iterrows():
        base_id = row['person_id']
        base_fn = row['first_nm']
        base_ln = row['last_nm']
        base_email = row['email_address']
        # base_domain = base_email.split('@')[-1]
        base_mdm = row['mdm_person_id']
        base_bd = pd.to_datetime(row['birth_dt'])
        base_img = row['headshot_b64']
        
        def record_variation(b_id, b_fn, b_ln, b_email, b_bd, b_img, b_type):
            return {
                'a_id': base_id,
                'b_id': b_id,
                'a_first': base_fn,
                'b_first': b_fn,
                'a_last': base_ln,
                'b_last': b_ln,
                'a_birth': base_bd.strftime('%Y-%m-%d'),
                'b_birth': b_bd.strftime('%Y-%m-%d'),
                'a_email': base_email,
                'b_email': b_email,
                'a_mdm': base_mdm,
                'b_mdm': base_mdm,
                'a_img': base_img,
                'b_img': b_img,
                'match': 1,
                'type': b_type
            }

        # Variant 1: minor changes
        pairs.append(record_variation(
            base_id + 10, base_fn.lower(), base_ln.upper(),
            base_email, base_bd + timedelta(days=random.randint(-300, 300)),
            flip_img_b64(base_img), 'standard'
        ))

        # Variant 2: missing image
        pairs.append(record_variation(
            base_id + 20, base_fn, base_ln,
            base_email, base_bd, None, 'missing_img'
        ))

        # Variant 3: nickname
        canon = nickname_map.get(base_fn.lower())
        if canon and canon != base_fn.lower():
            pairs.append(record_variation(
                base_id + 30, canon, base_ln,
                base_email, base_bd, base_img, 'nickname'
            ))

        # Variant 4: name hybrid
        hybrid_fn = f"{base_fn}-{canon or base_fn}"
        pairs.append(record_variation(
            base_id + 40, hybrid_fn, base_ln,
            base_email, base_bd, base_img, 'hybrid'
        ))
        
        # Variant 5: rotated image
        pairs.append(record_variation(
            base_id + 50, base_fn, base_ln,
            base_email, base_bd, rotate_img_b64(base_img, angle=random.choice([10, -10, 15])), 'rotated'
        ))
        
        # Variant 6: rotated image 180 degrees
        pairs.append(record_variation(
            base_id + 60, base_fn, base_ln,
            base_email, base_bd, rotate_180_b64(base_img), 'rotated_180'
        ))
        

    # Negatives
    shuffled = df.sample(frac=1).reset_index(drop=True)
    for i in range(len(df)):
        row1 = df.iloc[i]
        row2 = shuffled.iloc[i]
        if row1['person_id'] != row2['person_id']:
            pairs.append({
                'a_id': row1['person_id'],
                'b_id': row2['person_id'],
                'a_first': row1['first_nm'],
                'b_first': row2['first_nm'],
                'a_last': row1['last_nm'],
                'b_last': row2['last_nm'],
                'a_birth': row1['birth_dt'],
                'b_birth': row2['birth_dt'],
                'a_email': row1['email_address'],
                'b_email': row2['email_address'],
                # 'a_domain': row1['email_address'].split('@')[-1],
                # 'b_domain': row2['email_address'].split('@')[-1],
                'a_mdm': row1['mdm_person_id'],
                'b_mdm': row2['mdm_person_id'],
                'a_img': row1['headshot_b64'],
                'b_img': row2['headshot_b64'],
                'match': 0,
                'type': 'random'
            })

    return pd.DataFrame(pairs)

df_people, nickname_map = fetch_data()
print("✅ Fetched data from database")
df_pairs = generate_training_data(df_people, nickname_map)
print("✅ Generated training data")
df_pairs.to_csv("training_pairs_expanded.csv", index=False)
print("✅ Saved training_pairs_expanded.csv")
