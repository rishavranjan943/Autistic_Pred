import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os
from skimage.feature import hog

IMAGE_SIZE = (128, 128) 
MODEL_DIR = 'model'  


def extract_hog_features(image):
    # Resize the image to 128x128 and convert to grayscale
    image = image.resize(IMAGE_SIZE).convert('L')  # Convert to grayscale
    image_np = np.array(image)

    # Extract HOG features
    features = hog(image_np,
                   orientations=9,          # Gradient orientations
                   pixels_per_cell=(8, 8),  # Cell size for feature extraction
                   cells_per_block=(2, 2),  # Block size
                   block_norm='L2-Hys')     # Block normalization

    # If the number of features is greater than 7056, truncate to 7056 features
    if len(features) > 7056:
        features = features[:7056]

    # Return as a 2D array with 7056 features (shape: 1 x 7056)
    return features.reshape(1, -1)

# Streamlit UI
st.title("🧠 Autism Detection from Face Image")

# List model files from models directory
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

if not model_files:
    st.error("⚠️ No model files found in the 'models/' directory.")
else:
    selected_model = st.selectbox("Select a model file", model_files)

    # Load the selected model
    model_path = os.path.join(MODEL_DIR, selected_model)
    model = joblib.load(model_path)

    # File uploader for face image
    uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner('Extracting features and making prediction...'):
            try:
                # Extract HOG features
                features = extract_hog_features(image)
                
                # Display HOG feature length for debugging
                st.write(f"HOG feature length: {features.shape[1]}")

                # Make prediction using the model
                prediction = model.predict(features)[0]

                if prediction == 1:
                    st.success("🔴 The model predicts **Autistic**.")
                else:
                    st.success("🟢 The model predicts **Non-Autistic**.")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
