import pandas as pd
import numpy as np
from rapidfuzz.fuzz import partial_ratio
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import base64
import torch

# Load CLIP once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

def text_similarity(a: str, b: str) -> float:
    return partial_ratio(str(a).lower(), str(b).lower()) / 100.0

def b64_to_clip_embedding(b64: str):
    try:
        image = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features[0].numpy()
    except Exception:
        return np.zeros(512)  # fallback for invalid image

def extract_features(df: pd.DataFrame):
    features = pd.DataFrame()

    features['first_name_sim'] = [text_similarity(a, b) for a, b in zip(df['a_first'], df['b_first'])]
    features['last_name_sim'] = [text_similarity(a, b) for a, b in zip(df['a_last'], df['b_last'])]
    features['birthdate_match'] = (df['a_birth'] == df['b_birth']).astype(int)
    features['email_match'] = (df['a_email'].str.lower() == df['b_email'].str.lower()).astype(int)
    features['mdm_match'] = (df['a_mdm'] == df['b_mdm']).astype(int)

    # Image similarity via CLIP
    a_embeddings = [b64_to_clip_embedding(b64) for b64 in df['a_img']]
    b_embeddings = [b64_to_clip_embedding(b64) for b64 in df['b_img']]
    features['image_sim'] = [
        cosine_similarity([a], [b])[0][0] if np.any(a) and np.any(b) else 0.0
        for a, b in zip(a_embeddings, b_embeddings)
    ]

    # Return with optional target
    target = df['match'].astype(int) if 'match' in df.columns else None
    return features, target
