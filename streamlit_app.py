"""
FasalGuard - online demo (Streamlit Community Cloud)
Upload a leaf photo -> CNN diagnosis + treatment advice.
Same model, classes, treatments and safety rule as predict.py.
"""
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160

st.set_page_config(page_title="FasalGuard - Crop Disease Detection", page_icon="🌿")


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


st.title("🌿 FasalGuard - Crop Disease Detection")
st.write(
    "Deep-learning (CNN / MobileNetV2) demo trained on the PlantVillage dataset - "
    "**91.41% test accuracy** across 11 classes of corn, potato, tomato and bell pepper leaves. "
    "Upload a leaf photo and get the diagnosis plus treatment advice."
)

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded leaf", width=300)
    result = diagnose(image)
    st.subheader(f"Diagnosis: {result['diagnosis']}  ({result['confidence']:.1f}% confidence)")
    st.write(f"**Status:** {result['status']}")
    st.write(f"**Treatment advice:** {result['treatment']}")
    st.write(f"**Prevention:** {result['prevention']}")
    if result["low_confidence"]:
        st.warning("NOTE: Confidence is low - please also consult an agriculture expert.")
