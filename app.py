import streamlit as st
from PIL import Image
import numpy as np
import joblib
import os
from skimage.feature import hog
from sklearn.preprocessing import MinMaxScaler

IMAGE_SIZE = (128, 128) 
MODEL_DIR = 'model'  

# Load models
try:
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm.pkl'))
    rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf.pkl'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb.pkl'))
    knn_model = joblib.load(os.path.join(MODEL_DIR, 'knn.pkl'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

def extract_hog_features(image):
    image = image.resize(IMAGE_SIZE).convert('L')
    image_np = np.array(image)

    features = hog(image_np,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')

    if len(features) > 7056:
        features = features[:7056]

    return features.reshape(1, -1)

# UI
st.title("🧠 Autism Detection from Face Image (Ensemble Prediction)")

uploaded_file = st.file_uploader("Upload a face image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    with st.spinner('Extracting features and making prediction...'):
        try:
            features = extract_hog_features(image)

            # Get probabilities or decision scores
            knn_prob = knn_model.predict_proba(features)[0][1]
            rf_prob = rf_model.predict_proba(features)[0][1]
            xgb_prob = xgb_model.predict_proba(features)[0][1]

            # SVM: use decision_function then normalize
            svm_score = svm_model.decision_function(features)[0]
            # Normalize to [0, 1] using min-max scaler
            scaler = MinMaxScaler()
            svm_normalized = scaler.fit_transform([[svm_score], [0]])[0][0]

            # Weighted sum
            final_score = (
                0.24 * knn_prob +
                0.20 * rf_prob +
                0.32 * xgb_prob +
                0.20 * svm_normalized
            )

            prediction = 1 if final_score >= 0.5 else 0

            # Display intermediate values
            # st.write(f"🔍 KNN Probability (0.24): {knn_prob:.2f}")
            # st.write(f"🌲 RF Probability (0.20): {rf_prob:.2f}")
            # st.write(f"⚡ XGB Probability (0.32): {xgb_prob:.2f}")
            # st.write(f"📈 SVM Raw Score: {svm_score:.2f} → Normalized (0.20): {svm_normalized:.2f}")
            # st.write(f"📊 Final Weighted Score: {final_score:.2f}")

            if prediction == 1:
                st.success("🔴 The ensemble model predicts **Autistic**.")
            else:
                st.success("🟢 The ensemble model predicts **Non-Autistic**.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
