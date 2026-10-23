"""
FasalGuard - online demo (Hugging Face Spaces / Gradio)
Upload a leaf photo -> CNN diagnosis + treatment advice.
Same model, classes, treatments and safety rule as predict.py.
"""
import json

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 160

model = tf.keras.models.load_model("fasalguard_model.keras")
with open("classes.json") as f:
    meta = json.load(f)
CLASSES, DISPLAY = meta["classes"], meta["display"]
with open("treatments.json") as f:
    TREATMENTS = json.load(f)


def diagnose(image: Image.Image) -> str:
    if image is None:
        return "Please upload a leaf photo first."
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    probs = model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx]) * 100
    folder = CLASSES[idx]
    info = TREATMENTS[folder]

    lines = [
        f"Diagnosis (CNN) : {DISPLAY[folder]}  ({conf:.1f}% confidence)",
        f"Status          : {info['status']}",
        f"Treatment advice: {info['treatment']}",
        f"Prevention      : {info['prevention']}",
    ]
    if conf < 70:
        lines.append("NOTE: Confidence is low - please also consult an agriculture expert.")
    return "\n".join(lines)


demo = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="pil", label="Upload a leaf photo (corn, potato, tomato or bell pepper)"),
    outputs=gr.Textbox(label="FasalGuard diagnosis", lines=8),
    title="FasalGuard - Crop Disease Detection",
    description=(
        "Deep-learning (CNN / MobileNetV2) demo trained on the PlantVillage dataset - "
        "91.41% test accuracy across 11 classes of corn, potato, tomato and bell pepper leaves. "
        "Upload a leaf photo and get the diagnosis plus treatment advice."
    ),
    examples=[["examples/diseased_potato.jpg"], ["examples/healthy_tomato.jpg"]],
)

if __name__ == "__main__":
    demo.launch()
