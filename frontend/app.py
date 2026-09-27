import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import joblib
import numpy as np
from PIL import Image
import json

st.title("Fashion Recommender System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

#MODEL IMPORT
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fashionModel.keras"
    )

model = load_model()
knn = joblib.load("knn_model.pkl")
with open("image_paths.json", "r") as f:
    image_paths = json.load(f)


if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    embedding = model.predict(img_array, verbose=0)

    distances, indices = knn.kneighbors(embedding)

    nearest_paths = [image_paths[i] for i in indices[0]]

    cols = st.columns(5)

    for i, path in enumerate(nearest_paths):
        with cols[i]:
            st.image(path, use_container_width=True)