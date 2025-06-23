# Simple script to download fake faces from "thispersondoesnotexist.com"
import os
import time
import requests
from tqdm import tqdm

# Configuration
SAVE_DIR = "faces"
NUM_IMAGES = 1500
WAIT_SECONDS = 3
BASE_URL = "https://thispersondoesnotexist.com/"

# Fake a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36"
}

# Create local directory
os.makedirs(SAVE_DIR, exist_ok=True)

def download_face(idx):
    filename = os.path.join(SAVE_DIR, f"face_{idx:04}.jpg")
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"⚠️ Failed to download image {idx}: {e}")
        return False

print(f"📥 Starting download of {NUM_IMAGES} fake faces to '{SAVE_DIR}'...")

success_count = 0
failures = 0

for i in tqdm(range(NUM_IMAGES)):
    filepath = os.path.join(SAVE_DIR, f"face_{i:04}.jpg")
    if os.path.exists(filepath):
        continue
    if download_face(i):
        success_count += 1
    else:
        failures += 1
    time.sleep(WAIT_SECONDS)

print(f"\n✅ Downloaded {success_count} images.")
if failures:
    print(f"❌ {failures} failed downloads. You may retry those.")
