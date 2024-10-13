import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('ai_vs_nonai_model.pth', map_location=device)
model.to(device)
model.eval()

# Define image transformations
_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['AI', 'Non-AI']  # Adjust as per your dataset

# Streamlit app code
st.title("AI vs Non-AI Image Classifier")

st.write("Upload an image, and the model will classify it as 'AI' or 'Non-AI'.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image_tensor = _transform(image).unsqueeze(0).to(device)
    
    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        prediction = class_names[predicted_class.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted_class.item()] * 100

    # Display the prediction
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}%")
