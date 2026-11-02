import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PawVision AI | Animal Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background:
        radial-gradient(circle at 10% 20%, #19264a 0%, transparent 35%),
        radial-gradient(circle at 90% 80%, #301b46 0%, transparent 35%),
        #080d1a;
    color: #ffffff;
}

/* Hide Streamlit Header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main Container */
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

/* Animated Heading */
.hero-title {
    text-align: center;
    font-size: 65px;
    font-weight: 900;
    background: linear-gradient(
        90deg, #00f5ff, #a78bfa, #ff70c5, #00f5ff
    );
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient 6s linear infinite;
}

@keyframes gradient {
    0% { background-position: 0%; }
    100% { background-position: 300%; }
}

.hero-subtitle {
    text-align: center;
    color: #a5b4cc;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Glass Cards */
.glass-card {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 25px;
    backdrop-filter: blur(20px);
    box-shadow: 0 15px 45px rgba(0,0,0,0.25);
    animation: fadeUp 0.8s ease;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Section Heading */
.section-title {
    color: #e2e8f0;
    font-size: 23px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Upload Area */
[data-testid="stFileUploader"] {
    background: rgba(0,245,255,0.035);
    border: 2px dashed #596b96;
    border-radius: 18px;
    padding: 15px;
    transition: 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00f5ff;
    box-shadow: 0 0 20px rgba(0,245,255,0.15);
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 13px;
    padding: 15px;
    color: #07111f;
    font-size: 17px;
    font-weight: 800;
    background: linear-gradient(90deg, #00f5ff, #8b9cff);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 25px rgba(0,245,255,0.4);
    color: #07111f;
}

/* Result Card */
.result-card {
    background: linear-gradient(
        135deg,
        rgba(0,245,255,0.12),
        rgba(167,139,250,0.12)
    );
    border: 1px solid rgba(0,245,255,0.3);
    border-radius: 22px;
    padding: 30px;
    text-align: center;
    animation: resultAppear 0.8s ease;
}

@keyframes resultAppear {
    0% {
        opacity: 0;
        transform: scale(0.85);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

.result-label {
    color: #9caec9;
    font-size: 15px;
    letter-spacing: 3px;
}

.result-animal {
    font-size: 52px;
    font-weight: 900;
    color: #ffffff;
    margin: 12px 0;
}

.confidence {
    color: #00f5ff;
    font-size: 27px;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align: center;
    color: #71809c;
    padding: 35px 0 10px;
    font-size: 13px;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():
    import os
    model_path = os.path.join(os.path.dirname(__file__), "models", "dog_cat_cnn.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "dog_cat_cnn.keras")
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div style="text-align:center; font-size:45px;">
    🐾
</div>

<div class="hero-title">
    PawVision AI
</div>

<div class="hero-subtitle">
    Intelligent Animal Recognition Powered by Deep Learning
    <br>
    Upload an image and let CNN discover what's inside.
</div>
""", unsafe_allow_html=True)


# ==========================================
# MODEL INFORMATION
# ==========================================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("AI Architecture", "CNN")

with c2:
    st.metric("Image Resolution", "256 × 256")

with c3:
    st.metric("Classification", "Binary")


st.write("")


# ==========================================
# MAIN APPLICATION
# ==========================================

left, right = st.columns([1, 1], gap="large")

with left:

    st.markdown("""<div class="glass-card">
<div class="section-title">📤 Upload Your Image</div>
<p style="color:#9caec9;">Choose a picture of a cat or dog to begin AI analysis.</p>
</div>""", unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your Uploaded Image",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {image.width} × {image.height}"
        )

    else:

        st.markdown("""<div class="glass-card" style="text-align:center;">
<div style="font-size:65px;">🖼️</div>
<h3 style="color:#e2e8f0;">Waiting for an image</h3>
<p style="color:#899bb8;">Upload a JPG, JPEG or PNG image to start.</p>
</div>""", unsafe_allow_html=True)


with right:

    st.markdown("""<div class="glass-card">
<div class="section-title">🧠 AI Prediction Engine</div>
<p style="color:#9caec9;">Your trained CNN model analyzes the uploaded image and predicts the animal category.</p>
</div>""", unsafe_allow_html=True)

    st.write("")

    if uploaded_file:

        predict_button = st.button(
            "✨ ANALYZE IMAGE",
            use_container_width=True
        )

        if predict_button:

            with st.spinner("CNN is analyzing your image..."):

                # Image preprocessing
                img = image.resize((256, 256))

                img_array = np.array(img).astype("float32")

                img_array = img_array / 255.0

                img_array = np.expand_dims(
                    img_array, axis=0
                )

                # CNN Prediction
                prediction = model.predict(
                    img_array,
                    verbose=0
                )[0][0]

                # Class mapping:
                # 0 = Cats
                # 1 = Dogs

                if prediction < 0.5:

                    animal = "CAT"
                    emoji = "🐱"
                    confidence = (1 - prediction) * 100

                else:

                    animal = "DOG"
                    emoji = "🐶"
                    confidence = prediction * 100

                time.sleep(0.5)

            st.markdown(f"""<div class="result-card">
<div class="result-label">AI DETECTION COMPLETE</div>
<div style="font-size:85px; margin: 10px 0;">{emoji}</div>
<div class="result-animal">{animal}</div>
<div class="confidence">{confidence:.2f}%</div>
<p style="color:#9caec9; margin-top: 8px;">Model predicted class: {animal}</p>
</div>""", unsafe_allow_html=True)

            st.write("")

            st.markdown("### 📊 Prediction Confidence")

            st.progress(
                min(1.0, max(0.0, float(confidence / 100)))
            )

            st.caption(
                f"Raw CNN output: {float(prediction):.6f}"
            )

            st.success("Image analysis completed successfully!")


    else:

        st.markdown("""<div class="glass-card" style="text-align:center;">
<div style="font-size:65px;">🤖</div>
<h3 style="color:#e2e8f0;">AI Engine Ready</h3>
<p style="color:#899bb8;">Upload an image to activate the prediction engine.</p>
</div>""", unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================
# ==========================================
# FOOTER + CNN MODEL INFORMATION POPUP
# ==========================================

st.markdown("""
<style>
.footer-container {
    text-align: center;
    padding: 30px 10px;
    margin-top: 50px;
    border-top: 1px solid rgba(255,255,255,0.15);
}

.footer-title {
    font-size: 25px;
    font-weight: 800;
    background: linear-gradient(90deg,#00f5ff,#a78bfa,#ff70c5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.footer-text {
    color: #9caec9;
    font-size: 13px;
}

</style>

<div class="footer-container">
    <div class="footer-title">🐾 PawVision AI</div>
    <p class="footer-text">
        Intelligent Animal Classification using Convolutional Neural Networks
    </p>
    <p class="footer-text">
        Built with Python | TensorFlow | Keras | Streamlit
    </p>
    <p class="footer-text">
        Deep Learning Project • Dogs vs Cats Classification
    </p>
</div>
""", unsafe_allow_html=True)


# About CNN Button
if st.button("🧠 Explore CNN Architecture & Model Details"):

    @st.dialog("🧠 About PawVision AI — CNN Model")
    def show_model_details():

        st.markdown("### 🎯 Project Overview")
        st.write("""
        PawVision AI is a binary image classification application
        that uses a Convolutional Neural Network (CNN) to identify
        whether an uploaded image contains a cat or a dog.
        """)

        st.divider()

        st.markdown("### 🏗️ CNN Architecture")

        architecture = [
            ("Input Layer", "256 × 256 × 3"),
            ("Conv2D Block 1", "32 filters, 3×3, ReLU"),
            ("Batch Normalization", "Normalization"),
            ("Max Pooling 1", "2×2"),
            ("Conv2D Block 2", "64 filters, 3×3, ReLU"),
            ("Batch Normalization", "Normalization"),
            ("Max Pooling 2", "2×2"),
            ("Conv2D Block 3", "128 filters, 3×3, ReLU"),
            ("Batch Normalization", "Normalization"),
            ("Max Pooling 3", "2×2"),
            ("Flatten", "Feature extraction"),
            ("Dense Layer 1", "128 neurons, ReLU"),
            ("Dropout", "0.1"),
            ("Dense Layer 2", "64 neurons, ReLU"),
            ("Dropout", "0.1"),
            ("Output Layer", "1 neuron, Sigmoid")
        ]

        for layer, details in architecture:
            st.markdown(f"**{layer}:** {details}")

        st.divider()

        st.markdown("### ⚙️ Training Configuration")

        st.write("""
        - Framework: TensorFlow / Keras
        - Optimizer: Adam
        - Loss Function: Binary Crossentropy
        - Evaluation Metric: Accuracy
        - Batch Size: 32
        - Epochs: 10
        - Image Resolution: 256 × 256
        - Normalization: Pixel values divided by 255
        - Dataset: Dogs vs Cats (Kaggle)
        """)

        st.divider()

        st.markdown("### 📊 Model Performance")

        col1, col2 = st.columns(2)

        col1.metric("Training Accuracy", "93.77%")
        col2.metric("Best Validation Accuracy", "82.58%")

        st.info("""
        The model achieved 82.58% validation accuracy at Epoch 9.
        The difference between training and validation performance
        indicates signs of overfitting.
        """)

        st.divider()

        st.markdown("### 🔬 How CNN Works")

        st.write("""
        1. Input: The uploaded image is resized to 256×256 pixels.
        2. Convolution: Extracts image features such as edges,
           textures and shapes.
        3. Batch Normalization: Stabilizes network training.
        4. Max Pooling: Reduces spatial dimensions.
        5. Flatten: Converts feature maps into a one-dimensional vector.
        6. Dense Layers: Learn classification patterns.
        7. Sigmoid: Produces a probability between 0 and 1.
        8. Prediction: Classifies the image as Cat or Dog.
        """)

        st.divider()

        st.markdown("### 🧰 Technologies Used")

        st.write("""
        Python • TensorFlow • Keras • NumPy • Pillow • Streamlit
        """)

        st.caption("PawVision AI | CNN Image Classification Project")

    show_model_details()