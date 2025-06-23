# embedding_cache.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import base64
from io import BytesIO

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=True
)
clip_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

cache = {}

def encode_image_b64_to_vector(b64_img):
    if b64_img in cache:
        return cache[b64_img]

    image = Image.open(BytesIO(base64.b64decode(b64_img))).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(clip_model.device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)[0]
        emb = emb / emb.norm()
    vec = emb.cpu().numpy().tolist()
    cache[b64_img] = vec
    return vec
