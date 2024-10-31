import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the TensorFlow model once
MODEL = tf.keras.models.load_model(r"C:\Users\sanir\OneDrive\Documents\Plant Disease detection\models\potatoes.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function for model prediction
def model_prediction(image):
    image = image.resize((128, 128))  # Resize to match model input
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = MODEL.predict(input_arr)
    return CLASS_NAMES[np.argmax(predictions)], np.max(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿")
    
    # Set the image path
    image_path = r"C:\Users\sanir\OneDrive\Documents\Plant Disease detection\images\plant.png"

    # Check if the image exists before loading
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.error("Image file not found. Please check the file path.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    """)

# About Project
elif app_mode == "About":
    st.header("🔍 About")
    st.markdown("""
                #### About Dataset
                This dataset consists of 2152 files belonging to 3 classes:
                - **Potato___Early_blight**
                - **Potato___Late_blight**
                - **Potato___healthy**
                
                The dataset has been created using offline augmentation from the original dataset. The original dataset can be found on the Kaggle.
                This dataset is specifically designed to help in the identification of diseases affecting potato plants.
                
                ### Visualizations
                - Here are some sample images from each class:
                """)
    
    # Sample Images (Add your image paths here)
    sample_images = {
         "Early Blight": r"C:\Users\sanir\OneDrive\Documents\Plant Disease detection\images\earlyblight.jpg",
    "Late Blight": r"C:\Users\sanir\OneDrive\Documents\Plant Disease detection\images\lateblight.jpg",
    "Healthy": r"C:\Users\sanir\OneDrive\Documents\Plant Disease detection\images\healthy.jpg"
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(sample_images["Early Blight"], caption="Early Blight", use_column_width=True)
    with col2:
        st.image(sample_images["Late Blight"], caption="Late Blight", use_column_width=True)
    with col3:
        st.image(sample_images["Healthy"], caption="Healthy", use_column_width=True)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🌱 Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()  # Show loading animation
            class_name, confidence = model_prediction(image)
            st.success(f"Model predicts: {class_name} with confidence {confidence:.2f}")

            # Display prediction confidence as a bar chart
            predictions = MODEL.predict(np.expand_dims(image.resize((128, 128)), axis=0))
            plt.bar(CLASS_NAMES, predictions[0], color=['lightcoral', 'lightblue', 'lightgreen'])
            plt.xlabel('Classes')
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence')
            st.pyplot(plt)

            # Display recommendation based on prediction
            if class_name == "Early Blight":
                st.warning("Recommendation: Apply appropriate fungicide and improve air circulation.")
            elif class_name == "Late Blight":
                st.warning("Recommendation: Remove infected plants and consider resistant varieties.")
            else:
                st.success("Your plant is healthy! Keep up the good care!")
