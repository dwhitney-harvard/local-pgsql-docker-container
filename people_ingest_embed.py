import os
import psycopg2
import random
import base64
from datetime import datetime, timedelta
from faker import Faker
from faker.providers import BaseProvider
from PIL import Image
from embedding_cache import encode_image_b64_to_vector

FILE_RANGE = 1000

# fake = Faker()
fake = Faker([
    'en_US',  # English
    'de_DE',  # German
    'fr_FR',  # French
    'es_ES',  # Spanish
    # 'zh_CN',  # Chinese
    # 'ja_JP',  # Japanese
    'lv_LV',  # Latvian
    'uk_UA',  # Ukrainian
    'ar_AE',  # Arabic (UAE)
])

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

HEADSHOT_DIR = "faces"

start_time = datetime.now()
print("⏱️ Starting ingestion process @ %s..." % start_time.strftime('%Y-%m-%d %H:%M:%S'))

conn = psycopg2.connect(**DB)
cur = conn.cursor()

print("✅ Connected to PostgreSQL database")
print("✅ Using headshots from directory: %s" % HEADSHOT_DIR)
print("✅ Using %s files for random people generation" % FILE_RANGE)

def random_birth_date():
    start_date = datetime.strptime('1940-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
    delta = end_date - start_date
    return start_date + timedelta(days=random.randint(0, delta.days))

def generate_person():
    person_id = random.randint(10000, 999999)
    # Ensure unique person_id
    while cur.execute("SELECT 1 FROM people_with_faces WHERE person_id = %s", (person_id,)):
        person_id = random.randint(10000, 999999)
    
    first_nm = fake.first_name()
    last_nm = fake.last_name()
    birth_dt = random_birth_date().strftime('%Y-%m-%d')
    mdm_person_id = f"{random.randint(10000, 9999999)}"
    email = fake.email()
    headshot_file = random.choice(os.listdir(HEADSHOT_DIR))

    with open(os.path.join(HEADSHOT_DIR, headshot_file), "rb") as img:
        headshot_b64 = base64.b64encode(img.read()).decode("utf-8")

    embedding = encode_image_b64_to_vector(headshot_b64)

    return (person_id, first_nm, last_nm, birth_dt, mdm_person_id, email, headshot_b64, embedding)

cur.execute("""
    CREATE TABLE IF NOT EXISTS people_with_faces (
        id SERIAL PRIMARY KEY,
        person_id INT,
        first_nm TEXT,
        last_nm TEXT,
        birth_dt DATE,
        mdm_person_id BIGINT,
        email_address TEXT,
        headshot_b64 TEXT,
        face_embedding VECTOR(512)
    );
""")


people = [generate_person() for _ in range(FILE_RANGE)]
for p in people:
    cur.execute("""
        INSERT INTO people_with_faces
        (person_id, first_nm, last_nm, birth_dt, mdm_person_id, email_address, headshot_b64, face_embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, p)
    # print("✅ Ingested person: %s %s, ID: %s" % (p[0], p[1], p[3]))

conn.commit()
cur.close()
conn.close()
print("✅ Ingested %s people with embeddings" % FILE_RANGE)
print("⏱️ Finished ingestion process @ %s" % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("Total time taken: %s seconds" % (datetime.now() - start_time).total_seconds())
