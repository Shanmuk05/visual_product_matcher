# debug_query.py (auto-select first image if placeholder used)
from PIL import Image
import numpy as np
import os
from utils import extract_features_pil, load_features, find_similar_from_vector

# If you want to force a specific file, set it here:
QUERY = "images/your_query.jpg"   # leave as placeholder to auto-select

def find_first_image_in_dir(folder="images"):
    cand = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg",".jpeg",".png"))]
    return os.path.join(folder, cand[0]) if cand else None

def main():
    print("\n--- DEBUG QUERY TEST ---\n")

    print("Loading features_clip.pkl ...")
    feats = load_features("features_clip.pkl")
    print("Total features loaded:", len(feats))

    # if placeholder or missing, auto-select first image
    if not os.path.exists(QUERY):
        auto = find_first_image_in_dir("images")
        if auto is None:
            print("ERROR: No images found in images/ folder.")
            return
        print(f"Note: QUERY path '{QUERY}' not found. Auto-selecting '{auto}' for debug.")
        query_path = auto
    else:
        query_path = QUERY

    print("Using query image:", query_path)
    img = Image.open(query_path).convert("RGB")
    vec = extract_features_pil(img)

    print("\nQuery vector shape:", getattr(vec, "shape", None))
    print("Query L2 norm:", float(np.linalg.norm(vec)))
    print("Query min/max:", float(vec.min()), float(vec.max()))

    results = find_similar_from_vector(vec, feats, top_n=10)

    print("\n--- TOP 10 MATCHES ---")
    for fname, score in results:
        exists = os.path.exists(os.path.join("images", fname))
        print(f"{fname:40s}  {score:.4f}   FileExists: {exists}")

    print("\n--- END DEBUG ---\n")

if __name__ == "__main__":
    main()
