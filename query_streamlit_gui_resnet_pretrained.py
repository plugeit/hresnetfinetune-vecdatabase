import streamlit as st
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pymilvus import connections, Collection
import time

# Define the path for the saved model and connect to Milvus
MODEL_PATH = 'pytorch_resnet34.pth'
COLLECTION_NAME = 'image_embeddings'
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

# Connect to Milvus
def connect_to_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()  # Ensure the collection is loaded before querying
    return collection

# Load the ResNet-34 model and use it as an embedding generator
def load_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet34(pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    embedding_model = nn.Sequential(*list(model.children())[:-1])
    embedding_model.eval()
    embedding_model.to(device)
    return embedding_model, device

# Define image transformation
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Generate an embedding for the uploaded image
def generate_embedding(image, model, device):
    image = _transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        embedding = model(image).flatten()
    return embedding.cpu().numpy()

# Perform a search in Milvus for similar images
def search_similar_images(query_embedding, collection, top_n=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embeddings",
        param=search_params,
        limit=top_n,
        output_fields=["image_name"]
    )
    return results

# Streamlit GUI for querying images
def main():
    st.title("Image Search with Milvus")
    st.write("Upload an image to find similar images in the database.")

    # File uploader to upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Load the uploaded image and display it
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the model and connect to Milvus
        embedding_model, device = load_embedding_model()
        milvus_collection = connect_to_milvus()

        # Generate the embedding for the uploaded image
        query_embedding = generate_embedding(image, embedding_model, device)

        st.write("Searching for similar images...")
        start_time = time.time()

        # Search for similar images in Milvus
        search_results = search_similar_images(query_embedding, milvus_collection, top_n=5)
        
        end_time = time.time()
        st.write(f"Search completed in {end_time - start_time:.2f} seconds.")

        # Display the results
        if search_results:
            st.write(f"Top {len(search_results[0].ids)} similar images:")
            for match in search_results[0]:
                image_name = match.entity.get("image_name")
                retrieved_image = Image.open("combine/"+image_name)
                st.image(retrieved_image, caption=f"Similar Image: {image_name}", use_column_width=True)
                distance = match.distance
                st.write(f"Image: {image_name}, Distance: {distance:.4f}")
        else:
            st.write("No similar images found.")

if __name__ == "__main__":
    main()
