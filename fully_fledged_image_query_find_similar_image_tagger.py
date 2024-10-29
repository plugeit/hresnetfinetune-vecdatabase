import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from torchvision import models, transforms
import numpy as np
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility  # Added necessary imports
import time
import shelve
import uuid

global processor, caption_model, _transform, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
# Load the Processor and Model
SAVE_DIR = "uploaded_images"
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'image_embeddings'
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define image transformation and embedding generation
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

def generate_copyright_id():
    with shelve.open("cpids") as db:
        while True:
            uid=uuid.uuid4()
            try:
                db[uid]
            except Exception as e:
                print(e)
                return uid
    
def insert_new_image(main_image, raw_image, misinfo=False):
    # Path to the folder containing images
    image_folder_path = "combine"  # Replace with the path to your folder
    SAVE_DIR = "uploaded_images"
    os.makedirs(SAVE_DIR, exist_ok=True)  # Create the directory if it doesn't exist
    global processor, model
    # Load the Processor and Model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


    # Connect to Milvus
    # Function to create an index on the collection

    def generate_copyright_id():
        with shelve.open("cpids") as db:
            while True:
                uid=uuid.uuid4()
                try:
                    db[uid]
                except Exception as e:
                    print(e)
                    return uid

    def metadata_generator(image_main): #assuming image_main is preprocessed(image = Image.open(img_path).convert('RGB'))
        # Prepare the Inputs
        text = "an image of"
        inputs = processor(image_main, text, return_tensors="pt")

        # Generate the Caption
        output = model.generate(**inputs)
        caption=(f"Description : {processor.decode(output[0], skip_special_tokens=True)}")
        end_time=time.time()
        return caption

    def create_index(collection):
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        print("Index created on 'embeddings' field.")
    def connect_to_milvus():
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        collection = Collection(COLLECTION_NAME)
        collection.load()  # Ensure the collection is loaded before querying
        return collection

    # Load the ResNet-34 model and use it as an embedding generator
    def load_embedding_model_finetuned():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = models.resnet34(pretrained=False)  # Set pretrained=False to prevent downloading
        # Load the saved state dictionary from your local file
        model = torch.load('ai_vs_nonai_model_34.pth', map_location=device)
        embedding_model = nn.Sequential(*list(model.children())[:-1])
        embedding_model.eval()
        embedding_model.to(device)
        return embedding_model, device

    def load_embedding_model_pretrained():
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
            FieldSchema(name="image_name", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
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
    def process_folder_and_store_embeddings(img_path, model, device, collection, raw_image):
        embeddings = []
        image_names = []
        image_paths = [img_path]

        for idx, img_path in enumerate(image_paths):
            image = main_image
            embedding = generate_embedding(image, model, device)
            embeddings.append(embedding.astype(np.float32))
            #image_name = os.path.basename(img_path)  # Use the filename as the unique identifier
            uid = generate_copyright_id()
            copyright_id=str(uid)
            # Get the directory where the file was uploaded
            upload_dir = os.path.dirname(raw_image.name)
            # Construct the full file path
            file_name = os.path.basename(raw_image.name)
            file_path = os.path.join(upload_dir, file_name)
            metadata=metadata_generator(image)
            flags=[metadata]
            metadata=metadata.lower()
            st.write(metadata)
            image_metadata={"copyright_id":copyright_id,"image_name":file_path, "flags":flags, "platform_metadata":None}
            st.write(image_metadata)
            with shelve.open("cpids") as db:
                db[copyright_id] = image_metadata
            file_name=copyright_id+"."+raw_image.type.split("/")[-1]
            st.write(file_name)
            save_path = os.path.join(SAVE_DIR, file_name)  # Save as SAVE_DIR/filename.ext
            with open(save_path, "wb") as f:
                f.write(raw_image.getbuffer())
            image_names.append(str(image_metadata))
            print(f"done generating embedding for {img_path}")

        # Insert data into Milvus
        if embeddings:
            # Convert lists into NumPy arrays
            embeddings_np = np.vstack(embeddings)  # Stack into a 2D NumPy array of shape (num_images, 512
            start_time = time.time()
            # Insert data directly as a list of arrays into the Milvus collection
            collection.insert([image_names, embeddings_np])
            collection.flush()
            end_time = time.time()
            insertion_time = end_time - start_time
            st.write(f"Time taken to insert image: {insertion_time:.2f} seconds")
            create_index(collection)
        else:
            print("No embeddings were generated.")
    embedding_model, device = load_embedding_model_finetuned()
    milvus_collection = connect_to_milvus()
    # Process images from the folder and store them in Milvus
    uploaded_file = main_image
    process_folder_and_store_embeddings(main_image, embedding_model, device, milvus_collection, raw_image=raw_image)    
def check_flags(image_main, image_metadata, image_name, distance, raw_image):
    #if find flags like gun and fight in the similar image simply check for it in the original image
    if(image_metadata["flags"].count("misinformation")==1):
        #level A needs to be taken down immeditaely
        insert_new_image(image_main, misinfo=True, raw_image=raw_image)
    elif(image_metadata["flags"].count("nfsw")==1 or image_metadata["flags"][0].count("gun")>=1 or image_metadata["flags"][0].count("fight")>=1):
        #level B needs to be informed to platforms for moderation like blurring the image out etc
        insert_new_image(image_main, misinfo=False, raw_image=raw_image)
    else:
        #level C nothing needs to be done
        insert_new_image(image_main, misinfo=False, raw_image=raw_image)


def find_similar_image(image_main, raw_image):
    # Define the path for the saved model and connect to Milvus
    MODEL_PATH = 'ai_vs_nonai_model_34.pth'
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
        model = models.resnet34(pretrained=False)  # Set pretrained=False to prevent downloading
        # Load the saved state dictionary from your local file
        model = torch.load('ai_vs_nonai_model_34.pth', map_location=device)
        embedding_model = nn.Sequential(*list(model.children())[:-1])
        embedding_model.eval()
        embedding_model.to(device)
        return embedding_model, device

    # Perform a search in Milvus for similar images
    def search_similar_images(query_embedding, collection, top_n=1):
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

        # Load the model and connect to Milvus
        embedding_model, device = load_embedding_model()
        milvus_collection = connect_to_milvus()

        # Generate the embedding for the uploaded image
        query_embedding = generate_embedding(image_main, embedding_model, device)

        st.write("Searching for similar images...")
        start_time = time.time()

        # Search for similar images in Milvus
        search_results = search_similar_images(query_embedding, milvus_collection, top_n=1)
        
        end_time = time.time()
        st.write(f"Search completed in {end_time - start_time:.2f} seconds.")

        # Display the results
        if search_results:
            st.write(f"Top {len(search_results[0].ids)} similar images:")
            for match in search_results[0]:
                image_metadata=eval(match.entity.get("image_name"))
                st.write(image_metadata)
                image_name = image_metadata["image_name"]
                try:
                    retrieved_image = Image.open(image_name)
                except:
                    SAVE_DIR = "uploaded_images"
                    file_name_jpg=image_metadata["copyright_id"]+"."+"jpg"
                    file_name_png=image_metadata["copyright_id"]+"."+"png"
                    file_name_jpeg=image_metadata["copyright_id"]+"."+"jpeg"
                    save_path_jpg = os.path.join(SAVE_DIR, file_name_jpg)  # Save as SAVE_DIR/filename.ext
                    save_path_png = os.path.join(SAVE_DIR, file_name_png)  # Save as SAVE_DIR/filename.ext
                    save_path_jpeg = os.path.join(SAVE_DIR, file_name_jpeg)  # Save as SAVE_DIR/filename.ext
                    try:
                        retrieved_image = Image.open(save_path_jpg)
                    except:
                        try:
                            retrieved_image = Image.open(save_path_jpeg)
                        except:
                            try:
                                retrieved_image = Image.open(save_path_png)
                            except:
                                st.write("can't recover image")

                st.image(retrieved_image, caption=f"Similar Image: {image_name}", use_column_width=True)
                distance = match.distance
                if(int(distance==0)): #exact match found
                    st.write("exact_match_found")
                    st.write(f"copyright_id: {image_metadata["copyright_id"]}, Image: {image_name}, flags: {image_metadata["flags"]}, platform_metadata:{image_metadata["platform_metadata"]}, Distance: {distance:.4f}")
                else:
                    st.write(f"distance:{distance}")   
                    check_flags(image_main, image_metadata,image_name, distance, raw_image=raw_image)
        else:
            st.write("No similar images found.")
    main()


def classify_ai_nonai():
    # Load the trained model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('ai_vs_nonai_model_34.pth', map_location=device)
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
    st.title("Fully fledged query API")

    st.write("Upload an image and it will return an unique copright id and the metadata")

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

        if(predicted_class.item()==0 or predicted_class.item()==1):
            find_similar_image(image, raw_image=uploaded_file)

classify_ai_nonai()
