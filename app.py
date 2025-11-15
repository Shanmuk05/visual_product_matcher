import streamlit as st
from PIL import Image
import pandas as pd
import os
import time
from io import BytesIO
from utils import extract_features_pil, load_features, find_similar_from_vector, download_image_from_url

st.set_page_config(page_title="Visual Product Matcher", page_icon="🔎", layout="centered")

st.markdown("""
<style>
.stApp {
     background-image: url("C:/Users/Rishika Reddy D/Downloads/bg.jpg");
     background-size: cover;
     background-position: center;
     background-attachment: fixed;              
 }

.hero { 
    display:flex; 
    gap:28px; 
    align-items:center; 
    justify-content:center; 
    margin-bottom:20px; 
}
.hero-card { 
    background:white; 
    border-radius:14px; 
    box-shadow:0 10px 30px rgba(11,35,71,0.09); 
    padding:28px; 
    max-width:1000px;
    width:100%; 
    display:flex; 
    align-items:center;
}
.hero-text { flex:1; padding-right:20px; }
.hero-title { font-size:40px; font-weight:800; color:#06283D; margin:0 0 8px 0; }
.hero-sub { color:#355070; margin:0 0 12px 0; font-size:16px; }
.cta-btn { background:#1E90FF; color:white; padding:10px 18px; border-radius:8px; text-decoration:none; font-weight:600; }

.uploader-box { 
    background:white; 
    padding:18px; 
    border-radius:10px; 
    box-shadow:0 6px 16px rgba(11,35,71,0.06); 
}

.card { 
    background:white; 
    border-radius:10px; 
    padding:8px; 
    box-shadow:0 6px 18px rgba(11,35,71,0.06); 
    text-align:center; 
}
.card-title { font-size:13px; color:#06283D; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

banner_path = "images/banner.jpg"
hero_image = None
if os.path.exists(banner_path):
    try:
        hero_image = Image.open(banner_path).convert("RGB")
    except:
        hero_image = None

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.markdown('<div class="hero-card">', unsafe_allow_html=True)

st.markdown('<div class="hero-text">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🔎 Visual Product Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Find visually similar products instantly based on image similarity.</div>', unsafe_allow_html=True)
st.markdown('<a href="#upload" class="cta-btn">Start Matching →</a>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if hero_image:
    st.image(hero_image, width=340)
else:
    st.markdown("""
    <div style="width:340px; height:180px; display:flex; align-items:center; justify-content:center; 
         background:linear-gradient(90deg,#E9F2FF,#FFF); border-radius:10px;">
        <div style="text-align:center;">
            <div style="font-size:28px; color:#1E90FF; font-weight:700;">📸</div>
            <div style="color:#355070; margin-top:6px;">Upload your product image</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div id="upload"></div>', unsafe_allow_html=True)
st.markdown("### Upload or paste an image")

col1, col2 = st.columns(2)

uploaded = None
url = ""

with col1:
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="uploader-box">', unsafe_allow_html=True)
    url = st.text_input("Or paste image URL")
    st.markdown('</div>', unsafe_allow_html=True)

query_img = None

if uploaded:
    try:
        query_img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Unable to open uploaded image: {e}")

elif url:
    try:
        query_img = download_image_from_url(url)
    except Exception as e:
        st.error(f"Unable to download from URL: {e}")

if query_img is None:
    st.info("Upload or paste an image to start.")
else:
    st.markdown("### Query Image")
    st.image(query_img, width=350)

    st.markdown("### Searching for similar products...")
    with st.spinner("Processing..."):
        features = load_features("features_clip.pkl")
        vec = extract_features_pil(query_img)
        features = load_features("features_clip.pkl")

        import numpy as np
       


        results = find_similar_from_vector(vec, features, top_n=6)

    st.markdown("### Top Matches")
    cols = st.columns(3)

    for i, (fname, score) in enumerate(results):
        col = cols[i % 3]
        img_path = os.path.join("images", fname)
        caption = f"{fname} — {score:.3f}"

        if os.path.exists(img_path):
            col.markdown('<div class="card">', unsafe_allow_html=True)
            col.image(img_path, width=200)
            col.markdown(f'<div class="card-title">{caption}</div>', unsafe_allow_html=True)
            col.markdown('</div>', unsafe_allow_html=True)
        else:
            col.write(caption)
