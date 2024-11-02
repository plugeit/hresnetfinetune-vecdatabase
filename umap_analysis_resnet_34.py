import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility  # Added necessary imports
import matplotlib.pyplot as plt
from umap import UMAP  # This is the correct import
import matplotlib
import streamlit as st


# Path to the folder containing images
image_folder_path = "combine"  # Replace with the path to your folder

# visualize the data using umap/tsne

def visualize_embeddings(embeddings, labels):
    umap_reducer = UMAP(n_components=2, random_state=42)
    embeddings_2d = umap_reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    fake_indices = [i for i, l in enumerate(labels) if 'fake' in l.lower()]
    real_indices = [i for i, l in enumerate(labels) if 'real' in l.lower()]
    
    ax.scatter(embeddings_2d[fake_indices, 0], embeddings_2d[fake_indices, 1], label="Fake Images", alpha=0.6)
    ax.scatter(embeddings_2d[real_indices, 0], embeddings_2d[real_indices, 1], label="Real Images", alpha=0.6)
    
    ax.legend()
    ax.set_title('UMAP Visualization of Embeddings')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    
    st.pyplot(fig)

# Connect to Milvus
# Function to create an index on the collection
def create_index(collection):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="embeddings", index_params=index_params)
    print("Index created on 'embeddings' field.")
def connect_to_milvus():
    connections.connect("default", host="localhost", port="19530")
    collection_name = "image_embeddings"
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        print(f"Collection '{collection_name}' dropped.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

    create_collection()
    collection = Collection("image_embeddings")  # Replace "image_embeddings" with your collection name
    return collection

# Load the ResNet-34 model and use it as an embedding generator
def load_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet34(pretrained=False)  # Set pretrained=False to prevent downloading
    # Load the saved state dictionary from your local file
    state_dict = torch.load('pytorch_resnet34.pth', map_location=device)  # Ensure the path is correct
    model.load_state_dict(state_dict)  # Load the state dict into the model
    # Create the embedding model by removing the classification layer (last fully connected layer)
    embedding_model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
    # Set the model to evaluation mode and transfer it to the appropriate device
    embedding_model.eval()
    embedding_model.to(device)
    return embedding_model, device

# Define image transformation and embedding generation
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_collection():
    # Define fields for the schema
    collection_name = "image_embeddings"
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists. Loading existing collection.")
        return Collection(collection_name)
    fields = [
        FieldSchema(name="image_name", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=512)  # Adjust dim according to your model's output
    ]

    # Create the collection schema
    schema = CollectionSchema(fields, description="Collection for image embeddings")
    
    # Create the collection
    collection = Collection("image_embeddings", schema)
    return collection

def generate_embedding(image, model, device):
    image = _transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        embedding = model(image).flatten()
    return embedding.cpu().numpy()

# Generate embeddings for all images in a folder and store them in Milvus
def process_folder_and_store_embeddings(folder_path, model, device, collection):
    embeddings = []
    image_names = []
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for idx, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            embedding = generate_embedding(image, model, device)
            embeddings.append(embedding.astype(np.float32))
            image_name = os.path.basename(img_path)  # Use the filename as the unique identifier
            image_names.append(image_name)
            print(f"done generating embedding for {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Insert data into Milvus
    if embeddings:
        # Convert lists into NumPy arrays
        embeddings_np = np.vstack(embeddings)  # Stack into a 2D NumPy array of shape (num_images, 512
        # Insert data directly as a list of arrays into the Milvus collection
        collection.insert([image_names, embeddings_np])
        collection.flush()
        print(f"Inserted {len(embeddings)} image embeddings into the Milvus collection.")
        create_index(collection)
        visualize_embeddings(embeddings, image_names)
    else:
        print("No embeddings were generated.")


# Main function to run the process
if __name__ == "__main__":
    def main():
        st.title("Image Embedding Visualization")
        st.write("Processing images and generating embeddings...")

        # Load the model and connect to Milvus
        embedding_model, device = load_embedding_model()
        milvus_collection = connect_to_milvus()

        # Process images from the folder and store them in Milvus
        process_folder_and_store_embeddings(image_folder_path, embedding_model, device, milvus_collection)
    main()
