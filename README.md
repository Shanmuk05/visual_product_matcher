🔍 Visual Product Matcher (CLIP)

A Streamlit-based application that finds visually similar products using OpenAI CLIP.
Upload or paste an image, and the app instantly retrieves the most similar items from your product catalog based on image embeddings.

🚀 Features

Upload an image or enter an image URL

Extract embeddings using OpenAI CLIP (ViT-B/32)

Compare query image with catalog images

Display top matching products with similarity scores

Clean and customizable UI

Supports CPU/GPU

🧠 How It Works

Catalog images are processed using CLIP to generate 512-dimensional feature vectors.

These vectors are stored in features_clip.pkl.

When a query image is uploaded, CLIP extracts its vector.

Cosine similarity is used to find the closest product images.

Top matches are displayed with score and preview.

📁 Project Structure
visual_product_matcher/
│── app.py                     # Streamlit UI
│── utils.py                   # CLIP model & similarity functions
│── prepare_features_clip.py   # Generate product features
│── debug_query.py             # Debug similarity testing
│── features_clip.pkl          # Precomputed embeddings
│── images/                    # Product images
│── requirements.txt           # Dependencies

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<yourusername>/visual_product_matcher.git
cd visual_product_matcher

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Generate features for catalog images
python prepare_features_clip.py

4️⃣ Run the application
streamlit run app.py

🧪 Debugging Similarity

Run the debug script to inspect feature extraction and matches:

python debug_query.py

📸 Demo

(Add screenshots of your UI and matching results here)

🌟 Future Enhancements

Category-based filtering

Text-to-image search (CLIP dual encoder)

Advanced ranking using FAISS

Deployment on Streamlit Cloud / HuggingFace Spaces

🤝 Contributing

Pull requests and suggestions are welcome!

⭐ Support

If you like this project, please star the repository — it motivates further improvements!
