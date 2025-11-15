import os
import pickle
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)

def extract_features_pil(pil_img):
    pil_img = pil_img.convert("RGB")
    inputs = _processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        emb = _model.get_image_features(pixel_values)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb[0].cpu().numpy()

def load_features(pkl_path="features_clip.pkl"):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"{pkl_path} not found. Run prepare_features_clip.py first.")
    with open(pkl_path, "rb") as f:
        feats = pickle.load(f)
    return feats

def find_similar_from_vector(query_vec, features_dict, top_n=6):
    """
    Robust similarity search:
      - builds matrix from features_dict values (order kept by keys)
      - L2-normalizes stored features and query vector
      - computes cosine similarity via dot product
      - returns list of (filename, similarity) tuples sorted desc
    """
    # keys in stable order
    keys = list(features_dict.keys())
    if len(keys) == 0:
        return []

    # stack to matrix N x D
    mat = np.vstack([np.asarray(features_dict[k], dtype=np.float32) for k in keys])

    # normalize stored features (L2)
    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat_norms[mat_norms == 0] = 1.0
    mat_unit = mat / mat_norms

    # normalize query vector
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0 or np.isnan(q_norm):
        q_unit = q  # will yield low/undefined similarities but avoid crashing
    else:
        q_unit = q / q_norm

    # cosine similarities = dot(mat_unit, q_unit)
    sims = np.dot(mat_unit, q_unit).reshape(-1)

    # guard for NaNs
    sims = np.nan_to_num(sims, nan=-1.0)

    # sort descending
    idx_sorted = np.argsort(sims)[::-1]
    top_idx = idx_sorted[:top_n]

    results = [(keys[i], float(sims[i])) for i in top_idx]
    return results

def download_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")
