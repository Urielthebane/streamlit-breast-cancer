import streamlit as st
import numpy as np
import os

from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

# ---------------------------------------------------------
# Custom styling (Streamlit-safe CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
.card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}
.small-label {
    font-size: 0.75rem;
    color: #374151;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_cancer_predictor.h5")

# ---------------------------------------------------------
# Load model and scaler (cached)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    data = load_breast_cancer(as_frame=True)
    X = data.data
    scaler = StandardScaler()
    scaler.fit(X)
    return model, scaler, list(X.columns)

model, scaler, FEATURES = load_model_and_scaler()

# ---------------------------------------------------------
# Sample autofill data (your JS values)
# ---------------------------------------------------------
SAMPLE_DATA = [
    14.2, 20.1, 90.2, 600,
    0.1, 0.13, 0.12, 0.08, 0.18,
    0.06, 0.05, 0.05, 0.02, 0.03,
    0.1, 0.12, 0.11, 0.08, 0.18,
    16, 25, 105, 700, 0.12,
    0.15, 0.14, 0.1, 0.2,
    0.18, 0.07
]

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("Breast Cancer Prediction System")
st.caption("Artificial Neural Network for Tumor Classification")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("🔹 Autofill Sample Data"):
        for i, feature in enumerate(FEATURES):
            st.session_state[feature] = SAMPLE_DATA[i]

    with st.form("prediction_form"):
        inputs = []

        cols = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                value = st.number_input(
                    feature.replace("_", " ").title(),
                    value=st.session_state.get(feature, 0.0),
                    format="%.5f",
                    key=feature
                )
                inputs.append(value)

        submitted = st.form_submit_button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    input_array = np.array(inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prob = float(model.predict(input_scaled, verbose=0)[0][0])

    st.subheader("Prediction Result")

    if prob >= 0.5:
        st.success("🟢 Benign Tumor")
    else:
        st.error("🔴 Malignant Tumor")

    st.write(f"Prediction Probability: **{prob:.4f}**")

    # Confidence bar (Streamlit equivalent of animation)
    st.progress(int(prob * 100))

    st.caption("Higher values indicate higher confidence")

