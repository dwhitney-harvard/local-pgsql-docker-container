import psycopg2
import base64
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from time import time

# Set up the stopwatch
start_time = time()
print("⏱️ Starting backfill process...")

# Load CLIP model + processor
t0 = time()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("✅ Loaded CLIP model in %.2f seconds" % (time() - t0))

# Optional: enable GPU if available
t1 = time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
print("✅ Moved CLIP model to %s in %.2f seconds" % (device, time() - t1))

# DB connection info
DB = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mydb",
    "user": "postgres",
    "password": "mypassword"
}

def encode_clip_embedding(b64_img: str) -> list:
    image = Image.open(BytesIO(base64.b64decode(b64_img))).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)[0]
        emb = emb / emb.norm()  # cosine-normalized
    return emb.cpu().numpy().tolist()

def main():
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()

    # Fetch all records missing a face_embedding
    cur.execute("""
        SELECT person_id, headshot_b64
        FROM people_with_faces
        WHERE face_embedding IS NULL
    """)
    rows = cur.fetchall()

    print(f"🚀 Found {len(rows)} rows to backfill")

    for person_id, b64_img in tqdm(rows):
        try:
            emb = encode_clip_embedding(b64_img)
            cur.execute("""
                UPDATE people_with_faces
                SET face_embedding = %s
                WHERE person_id = %s
            """, (emb, person_id))
        except Exception as e:
            print(f"⚠️ Skipped person_id={person_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Embedding backfill complete in %.2f seconds" % (time() - start_time))

if __name__ == "__main__":
    main()
