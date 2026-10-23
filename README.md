---
title: FasalGuard - Crop Disease Detection
emoji: 🌿
colorFrom: green
colorTo: teal
sdk: gradio
app_file: app.py
python_version: "3.13"
pinned: false
---

# FasalGuard - Crop Disease Detection

Deep-learning (CNN / MobileNetV2 transfer learning) web demo: upload a leaf photo of
corn, potato, tomato or bell pepper and get the disease diagnosis with treatment and
prevention advice. Trained on a PlantVillage subset (11 classes, 1,320 images) -
91.41% test accuracy. If the model confidence is below 70%, the app advises consulting
a human agriculture expert instead of guessing.
