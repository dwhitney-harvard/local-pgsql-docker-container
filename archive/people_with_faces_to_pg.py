# This script generates a CSV file with fake person data and their corresponding base64 encoded images,
# and then inserts this data into a PostgreSQL database. It uses the Faker library to generate
# random names, birth dates, and email addresses. The images are downloaded from a local directory
# containing fake faces. The script also handles database connection and error management.
import os
import csv
import base64
import random
import psycopg2
from faker import Faker
from datetime import datetime

# Configuration
NUM_PEOPLE = 1000
FACES_DIR = "faces"
CSV_FILE = "people_with_faces.csv"

DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

NUM_PEOPLE = 1500
person_ids = random.sample(range(100_000, 3_000_000), NUM_PEOPLE)
mdm_ids    = random.sample(range(5_000_000, 9_000_000), NUM_PEOPLE)

fake = Faker()
face_files = sorted(os.listdir(FACES_DIR))[:NUM_PEOPLE]

# ✅ 1. Generate People + Base64 Images to CSV
with open(CSV_FILE, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['person_id', 'first_nm', 'last_nm', 'birth_dt', 'mdm_person_id', 'email_address', 'headshot_b64'])

    for i, face in enumerate(face_files):
        with open(os.path.join(FACES_DIR, face), "rb") as img_file:
            headshot_b64 = base64.b64encode(img_file.read()).decode("utf-8")

        person_id = person_ids[i]
        mdm_person_id = mdm_ids[i]
        
        first_nm = fake.first_name()
        last_nm = fake.last_name()
        birth_dt = fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d')
        email_address = fake.email()

        writer.writerow([person_id, first_nm, last_nm, birth_dt, mdm_person_id, email_address, headshot_b64])

print(f"✅ Saved {NUM_PEOPLE} people with faces to {CSV_FILE}")

# ✅ 2. Insert into PostgreSQL
try:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    # Create table if not exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS people_with_faces (
        id SERIAL PRIMARY KEY,
        person_id INT,
        first_nm TEXT,
        last_nm TEXT,
        birth_dt DATE,
        mdm_person_id BIGINT,
        email_address VARCHAR(320),
        headshot_b64 TEXT
    );
    """)

    # Load CSV into PostgreSQL
    with open(CSV_FILE, 'r') as f:
        next(f)  # Skip header
        cur.copy_expert("""
            COPY people_with_faces(person_id, first_nm, last_nm, birth_dt, mdm_person_id, email_address, headshot_b64)
            FROM STDIN WITH CSV
        """, f)

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Inserted people with faces into PostgreSQL.")

except Exception as e:
    print(f"❌ Database error: {e}")
