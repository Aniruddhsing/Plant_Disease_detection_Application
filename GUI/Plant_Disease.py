import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the TensorFlow model using a relative path
MODEL_PATH = "models/potatoesnew.keras"
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  # Stop the app if the model can't be loaded

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function for model prediction
def model_prediction(image):
    image = image.resize((256, 256))  # Resize to match model input
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
    image_path = os.path.join("..", "images", "plant.png")

    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.error("Image file not found. Please check the file path.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    This system uses a deep learning model to identify plant diseases based on images.
    Simply navigate to the "Disease Recognition" page to upload an image of a plant leaf,
    and the model will predict its health status.
    """)

# About Project
elif app_mode == "About":
    st.header("🔍 About")
    st.markdown("""
    #### About Dataset
    This application is designed to recognize diseases in potato plants.
    
    The dataset contains images of potato leaves categorized into three classes:
    - Early Blight
    - Late Blight
    - Healthy
    
    The model has been trained using TensorFlow and Keras to provide accurate predictions.
    """)

    # Sample Images
    sample_images = {
        "Early Blight": os.path.join("images", "earlyblight.jpg"),
        "Late Blight": os.path.join("images", "lateblight.jpg"),
        "Healthy": os.path.join("images", "healthy.jpg"),
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

        if st.button("Predict"):
            st.snow()  # Show loading animation
            class_name, confidence = model_prediction(image)
            st.success(f"Model predicts: {class_name} with confidence {confidence:.2f}")

            predictions = MODEL.predict(np.expand_dims(image.resize((256, 256)), axis=0))
            plt.clf()  # Clear the previous plot
            plt.figure(figsize=(10, 5))
            plt.bar(CLASS_NAMES, predictions[0], color=['lightcoral', 'lightblue', 'lightgreen'])
            plt.xlabel('Classes')
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence')
            st.pyplot(plt)

            if class_name == "Early Blight":
                st.warning("Recommendation: Apply appropriate fungicide and improve air circulation.")
            elif class_name == "Late Blight":
                st.warning("Recommendation: Remove infected plants and consider resistant varieties.")
            else:
                st.success("Your plant is healthy! Keep up the good care!")
