import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load Model
@st.cache_resource
def load_model():
    model_path = "deepfake_model.pth"  # Adjust based on model path
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Define Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Change size based on your model
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust based on training
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("Deepfake Detection Model")
st.write("Upload an image to check if it's real or deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Deepfake"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        label = "Deepfake" if prediction == 1 else "Real"
        st.write(f"Prediction: **{label}**")
