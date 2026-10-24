"""
FasalGuard - online demo (Streamlit Community Cloud)
Modern interface: upload a leaf photo -> CNN diagnosis + treatment advice.
Same model, classes, treatments and safety rule as predict.py.
"""
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160

st.set_page_config(
    page_title="FasalGuard - Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
)

# ---------- Modern theme ----------
st.markdown(
    """
    <style>
    .stApp { background: #f4faf4; }
    .banner {
        background: linear-gradient(90deg, #1b7a3d, #0e9594);
        color: white; padding: 26px 30px; border-radius: 18px;
        margin-bottom: 22px;
    }
    .banner h1 { margin: 0; font-size: 34px; }
    .banner p { margin: 6px 0 0 0; opacity: 0.92; font-size: 16px; }
    .card {
        background: white; border-radius: 16px; padding: 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 16px;
    }
    .card h3 { margin-top: 0; }
    .chip {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-weight: 600; font-size: 14px; color: white;
    }
    .chip-green { background: #1b7a3d; }
    .chip-red { background: #c0392b; }
    .conf { font-size: 40px; font-weight: 700; margin: 4px 0; }
    .muted { color: #5b6b5e; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("fasalguard_model.keras")
    with open("classes.json") as f:
        meta = json.load(f)
    with open("treatments.json") as f:
        treatments = json.load(f)
    return model, meta["classes"], meta["display"], treatments


model, CLASSES, DISPLAY, TREATMENTS = load_all()


def diagnose(image: Image.Image) -> dict:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx]) * 100
    folder = CLASSES[idx]
    info = TREATMENTS[folder]
    return {
        "diagnosis": DISPLAY[folder],
        "confidence": conf,
        "status": info["status"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "low_confidence": conf < 70,
    }


# ---------- Header ----------
st.markdown(
    """
    <div class="banner">
        <h1>🌿 FasalGuard</h1>
        <p>AI Crop Disease Detection — upload a leaf photo, get the diagnosis and treatment advice instantly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_upload, col_result = st.columns([1, 1.15])

with col_upload:
    st.markdown('<div class="card"><h3>📸 Upload a leaf photo</h3>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Corn, potato, tomato or bell pepper leaf (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded is not None:
        st.image(Image.open(uploaded), caption="Your leaf photo", width=320)
    else:
        st.markdown(
            '<p class="muted">Tip: a clear, well-lit photo of a single leaf gives the best result.</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with col_result:
    if uploaded is not None:
        result = diagnose(Image.open(uploaded))
        chip_class = "chip-green" if result["status"] == "Healthy" else "chip-red"
        st.markdown(
            f"""
            <div class="card">
                <h3>🔍 Diagnosis</h3>
                <div class="conf">{result['diagnosis']}</div>
                <span class="chip {chip_class}">{result['status']}</span>
                <p style="margin-top:12px;">Confidence: <b>{result['confidence']:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(result["confidence"]), 100))
        st.markdown(
            f"""
            <div class="card">
                <h3>💊 Treatment advice</h3>
                <p>{result['treatment']}</p>
            </div>
            <div class="card">
                <h3>🛡️ Prevention</h3>
                <p>{result['prevention']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if result["low_confidence"]:
            st.warning("NOTE: Confidence is low - please also consult an agriculture expert.")
    else:
        st.markdown(
            """
            <div class="card">
                <h3>🔍 Diagnosis</h3>
                <p class="muted">Upload a leaf photo on the left and the diagnosis will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <p class="muted" style="text-align:center; margin-top:10px;">
    Model: MobileNetV2 transfer learning (CNN) · 91.41% test accuracy · 11 classes ·
    Trained on a PlantVillage subset · If confidence is below 70%, the app advises consulting a human agriculture expert.
    </p>
    """,
    unsafe_allow_html=True,
)
