import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Load the fine-tuned model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torch.load('ai_vs_nonai_model.pth', map_location=device)

# Remove the final classification layer to generate embeddings
embedding_model = nn.Sequential(*list(model.children())[:-1])
embedding_model.to(device)
embedding_model.eval()

# Define image transformations (same as during training)
_transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app code
st.title("Image Embedding Generator")

st.write("Upload an image to generate its embedding using the fine-tuned model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

def generate_embedding(image, model, device):
    """Generate embeddings for the given image using the specified model."""
    # Transform the image
    image = _transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Generate the embedding
    with torch.no_grad():
        embedding = model(image)
    
    # Flatten the embedding to a 1D vector
    embedding = embedding.view(embedding.size(0), -1).cpu().numpy()

    return embedding

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Generating embedding...")

    # Generate and display the embedding
    embedding = generate_embedding(image, embedding_model, device)
    st.write("Embedding generated:")
    st.write(embedding)
    st.write(f"Embedding shape: {embedding.shape}")
