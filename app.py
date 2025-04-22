import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import CSRNet  # make sure model.py is in the same folder
import io

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet()
    model.load_state_dict(torch.load("weights.pth", map_location=device))  # replace with your weights path
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Preprocessing function
def preprocess(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # add batch dim
    return img.to(device)

# Inference function
def predict(img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze().cpu().numpy()
        count = np.sum(output)
    return output, count

# App UI
st.title("🧠 CSRNet - Crowd Counting")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

threshold = st.slider("Threshold for 'Crowded' status", 0.0, 2000.0, 100.0)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict
    img_tensor = preprocess(img)
    density_map, count = predict(img_tensor)

    # Display results
    st.subheader("🔢 Predicted Head Count:")
    st.write(f"**{count:.2f} people**")

    if count >= threshold:
        st.error(f"🚨 Area is **CROWDED** (Threshold: {threshold})")
    else:
        st.success(f"✅ Area is **NOT Crowded** (Threshold: {threshold})")

    st.subheader("🔥 Density Heatmap:")
    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')
    st.pyplot(fig)
