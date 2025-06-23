import base64
import pandas as pd
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import torch

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def decode_headshot_to_image(b64_string):
    image = Image.open(BytesIO(base64.b64decode(b64_string))).convert("RGB")
    return image

def get_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        return clip_model.get_image_features(**inputs).squeeze().numpy()

# Load people table
df = pd.read_csv("people_with_faces.csv")

# Compute and store embeddings
embeddings = []
for i, row in df.iterrows():
    img = decode_headshot_to_image(row['headshot_b64'])
    emb = get_clip_embedding(img)
    embeddings.append(emb)

df['embedding'] = embeddings
df.to_pickle("people_with_embeddings.pkl")  # faster load
print("✅ Saved image embeddings to people_with_embeddings.pkl")
