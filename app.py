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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stApp {
    background: transparent;
}
.card {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 20px 45px rgba(0,0,0,0.15);
    border: 1px solid #e5e7eb;
    margin: 1rem 0;
}
.feature-group {
    background: #f9fafb;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}
.feature-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
}
.prediction-box {
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 1.5rem 0;
}
.benign {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
}
.malignant {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    color: white;
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
# Feature groups for better organization
# ---------------------------------------------------------
FEATURE_GROUPS = {
    "Mean Values": [f for f in FEATURES if f.startswith("mean")],
    "Standard Error": [f for f in FEATURES if f.startswith("radius error") or "error" in f],
    "Worst Values": [f for f in FEATURES if f.startswith("worst")]
}

# ---------------------------------------------------------
# Sample data with correct feature count (30 features)
# ---------------------------------------------------------
SAMPLE_DATA = {
    'mean radius': 14.2, 'mean texture': 20.1, 'mean perimeter': 90.2, 
    'mean area': 600.0, 'mean smoothness': 0.1, 'mean compactness': 0.13,
    'mean concavity': 0.12, 'mean concave points': 0.08, 
    'mean symmetry': 0.18, 'mean fractal dimension': 0.06,
    
    'radius error': 0.5, 'texture error': 0.5, 'perimeter error': 2.0,
    'area error': 30.0, 'smoothness error': 0.01, 'compactness error': 0.03,
    'concavity error': 0.03, 'concave points error': 0.008,
    'symmetry error': 0.02, 'fractal dimension error': 0.003,
    
    'worst radius': 16.0, 'worst texture': 25.0, 'worst perimeter': 105.0,
    'worst area': 700.0, 'worst smoothness': 0.12, 'worst compactness': 0.15,
    'worst concavity': 0.14, 'worst concave points': 0.1,
    'worst symmetry': 0.2, 'worst fractal dimension': 0.07
}

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    for feature in FEATURES:
        if feature not in st.session_state:
            st.session_state[feature] = 0.0

# ---------------------------------------------------------
# UI Header
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🔬 Breast Cancer Prediction")
    st.markdown("### AI-Powered Tumor Classification System")
    st.caption("Using Artificial Neural Networks for accurate diagnosis support")

# ---------------------------------------------------------
# Sidebar with instructions
# ---------------------------------------------------------
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. **Enter tumor measurements** or use sample data
    2. **Click Predict** to analyze
    3. **Review results** and confidence score
    
    ---
    
    ### About the Features
    
    **Mean Values**: Average measurements of cell nuclei
    
    **Standard Error**: Variability in measurements
    
    **Worst Values**: Largest/worst measurements observed
    
    ---
    
    ⚠️ **Note**: This is a diagnostic support tool. Always consult with healthcare professionals.
    """)
    
    if st.button("🔄 Reset All Values", use_container_width=True):
        for feature in FEATURES:
            st.session_state[feature] = 0.0
        st.rerun()
    
    if st.button("✨ Load Sample Data", use_container_width=True):
        for feature in FEATURES:
            st.session_state[feature] = SAMPLE_DATA.get(feature, 0.0)
        st.rerun()

# ---------------------------------------------------------
# Main content
# ---------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

# Create tabs for different feature groups
tab1, tab2, tab3 = st.tabs(["📊 Mean Values", "📈 Standard Error", "⚠️ Worst Values"])

inputs = {}

with tab1:
    st.markdown("### Mean Measurements")
    cols = st.columns(3)
    mean_features = [f for f in FEATURES if f.startswith("mean")]
    for i, feature in enumerate(mean_features):
        with cols[i % 3]:
            # Use float format to avoid int warning
            value = st.number_input(
                feature.replace("mean ", "").replace("_", " ").title(),
                value=float(st.session_state.get(feature, 0.0)),
                format="%.5f",
                key=f"input_{feature}",
                help=f"Enter {feature.replace('_', ' ')}"
            )
            inputs[feature] = value

with tab2:
    st.markdown("### Standard Error Measurements")
    cols = st.columns(3)
    error_features = [f for f in FEATURES if "error" in f]
    for i, feature in enumerate(error_features):
        with cols[i % 3]:
            value = st.number_input(
                feature.replace(" error", "").replace("_", " ").title(),
                value=float(st.session_state.get(feature, 0.0)),
                format="%.5f",
                key=f"input_{feature}",
                help=f"Enter {feature.replace('_', ' ')}"
            )
            inputs[feature] = value

with tab3:
    st.markdown("### Worst Value Measurements")
    cols = st.columns(3)
    worst_features = [f for f in FEATURES if f.startswith("worst")]
    for i, feature in enumerate(worst_features):
        with cols[i % 3]:
            value = st.number_input(
                feature.replace("worst ", "").replace("_", " ").title(),
                value=float(st.session_state.get(feature, 0.0)),
                format="%.5f",
                key=f"input_{feature}",
                help=f"Enter {feature.replace('_', ' ')}"
            )
            inputs[feature] = value

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Predict", use_container_width=True, type="primary")

# ---------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------
if predict_button:
    # Ensure inputs are in the correct order matching FEATURES
    input_array = np.array([inputs.get(f, 0.0) for f in FEATURES]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prob = float(model.predict(input_scaled, verbose=0)[0][0])
    
    st.markdown("---")
    
    # Results section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if prob >= 0.5:
            st.markdown("""
            <div class="prediction-box benign">
                🟢 BENIGN TUMOR
            </div>
            """, unsafe_allow_html=True)
            st.success("The tumor characteristics suggest a benign (non-cancerous) classification.")
        else:
            st.markdown("""
            <div class="prediction-box malignant">
                🔴 MALIGNANT TUMOR
            </div>
            """, unsafe_allow_html=True)
            st.error("The tumor characteristics suggest a malignant (cancerous) classification.")
    
    with col2:
        st.metric(
            label="Confidence Score",
            value=f"{prob:.2%}",
            delta=f"{abs(prob - 0.5):.2%} from threshold"
        )
        
        st.progress(prob, text=f"Probability: {prob:.4f}")
        
        # Interpretation
        if prob > 0.7 or prob < 0.3:
            confidence_level = "High"
            confidence_color = "green" if prob > 0.5 else "red"
        elif prob > 0.6 or prob < 0.4:
            confidence_level = "Moderate"
            confidence_color = "orange"
        else:
            confidence_level = "Low"
            confidence_color = "gray"
        
        st.markdown(f"**Confidence Level**: :{confidence_color}[{confidence_level}]")
    
    # Additional information
    with st.expander("📊 View Input Summary"):
        st.json({k: float(v) for k, v in inputs.items()})

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("⚕️ This tool is for educational and research purposes only. Always consult healthcare professionals for medical decisions.")