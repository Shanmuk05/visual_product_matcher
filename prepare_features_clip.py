import os, pickle, torch, numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

IMAGE_DIR = "images"
OUT_PATH = "features_clip.pkl"
MODEL_NAME = "openai/clip-vit-base-patch32"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def extract_image_embedding(pil_image):
    inputs = processor(images=pil_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return image_embeds[0].cpu().numpy()

def main():
    feats = {}
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for fname in tqdm(files):
        path = os.path.join(IMAGE_DIR, fname)
        img = Image.open(path).convert("RGB")
        feats[fname] = extract_image_embedding(img)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(feats, f)
    print("Saved embeddings to", OUT_PATH)

if __name__ == "__main__":
    main()
