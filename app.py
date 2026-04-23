import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# ----------------------------
# Load Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("model/pneumonia_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# UI
# ----------------------------
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and get prediction")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    classes = ["Normal", "Pneumonia"]

    st.subheader("Prediction:")
    st.success(f"Result: {classes[pred.item()]}")