import streamlit as st
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from PIL import Image
import os
import time

def load_image(image_file):
    img = Image.open(image_file)
    return img

def image_similarity(image1, image2):
    try:
        # Convert images to RGB if they are not already in RGB format
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        
        # Convert images to numpy arrays and then to grayscale
        image1_gray = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
        
        # Resize images to the same size
        if image1_gray.shape != image2_gray.shape:
            image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))
        
        # Compute SSIM between the two images
        similarity_index, _ = compare_ssim(image1_gray, image2_gray, full=True)
        
        return similarity_index * 100  # Return similarity percentage
    except:
        st.write(f"error detected when comparing: {image2}\n")
        return 0

def compare_with_directory(uploaded_image, dir_path):
    # List all files in the directory
    image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Store similarity results and timing data
    similarity_results = []
    time_per_image = []
    
    # Progress bar
    progress_bar = st.progress(0)
    total_files = len(image_files)
    
    start_time = time.time()
    
    for idx, image_file in enumerate(image_files):
        # Load the image from the directory
        image_path = os.path.join(dir_path, image_file)
        img = Image.open(image_path)
        
        # Start timer for each image comparison
        image_start_time = time.time()
        
        # Compute similarity
        similarity = image_similarity(uploaded_image, img)
        similarity_results.append((image_file, similarity, img))
        
        # End timer for each image comparison
        image_end_time = time.time()
        time_taken = image_end_time - image_start_time
        time_per_image.append(time_taken)
        
        # Update progress bar
        progress_bar.progress((idx + 1) / total_files)
    
    # Sort results by similarity score in descending order
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate total time
    total_time = time.time() - start_time
    avg_time_per_image = sum(time_per_image) / len(time_per_image)
    
    # Return top 5 results and timing statistics
    return similarity_results[:5], total_time, avg_time_per_image, len(image_files)

def main():
    st.title("Image Similarity Comparator with Directory")
    
    # Upload one main image for comparison
    image_file = st.file_uploader("Upload the main image", type=["png", "jpg", "jpeg"])
    
    # Select directory containing images to compare
    dir_path = st.text_input("Enter the directory path containing images to compare:")
    
    if image_file and dir_path:
        # Load the uploaded image
        uploaded_image = load_image(image_file)
        
        st.image(uploaded_image, caption="Uploaded Image", width=300)
        
        # Compare the uploaded image with all images in the directory
        if st.button("Compare with Directory"):
            try:
                st.write("Starting the comparison process...")
                
                top_similar_images, total_time, avg_time_per_image, total_images = compare_with_directory(uploaded_image, dir_path)
                
                # Display the top 5 similar images
                st.subheader("Top 5 Most Similar Images")
                for image_name, similarity, img in top_similar_images:
                    st.image(img, caption=f"{image_name} - {similarity:.2f}% Similarity", width=300)
                
                # Show statistics
                st.subheader("Comparison Statistics")
                st.write(f"Total images compared: {total_images}")
                st.write(f"Total time taken: {total_time:.2f} seconds")
                st.write(f"Average time per image: {avg_time_per_image:.2f} seconds")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
if __name__ == "__main__":
    main()
