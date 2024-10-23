import streamlit as st
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from PIL import Image

def load_image(image_file):
    img = Image.open(image_file)
    return img

def image_similarity(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)
    
    # Resize images to the same size
    if image1_gray.shape != image2_gray.shape:
        image2_gray = cv2.resize(image2_gray, (image1_gray.shape[1], image1_gray.shape[0]))
    
    # Compute SSIM between the two images
    similarity_index, _ = compare_ssim(image1_gray, image2_gray, full=True)
    
    return similarity_index * 100  # Return similarity percentage

def main():
    st.title("Image Similarity Comparator")
    
    # Upload two images for comparison
    image_file1 = st.file_uploader("Upload the first image", type=["png", "jpg", "jpeg"])
    image_file2 = st.file_uploader("Upload the second image", type=["png", "jpg", "jpeg"])
    
    if image_file1 and image_file2:
        img1 = load_image(image_file1)
        img2 = load_image(image_file2)
        
        st.image([img1, img2], caption=["First Image", "Second Image"], width=300)
        
        if st.button("Compare Images"):
            similarity = image_similarity(img1, img2)
            st.success(f"The similarity between the images is {similarity:.2f}%")
    
if __name__ == "__main__":
    main()
