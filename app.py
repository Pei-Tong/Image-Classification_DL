import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model("best_model.h5")

# Define class labels
class_labels = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Streamlit application interface
st.title("Image Classifier")
st.write("Upload an image - Buildings, Forest, Glacier, Mountain, Sea or Street , to let the AI recognize its category!")


# User selects image resizing method
resize_option = st.radio(
    "Image Processing Method:",
    ["Resize with Padding (Maintain Aspect Ratio)", "Central Crop (Remove Edges)"]
)

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Load the image
    img = image.load_img(uploaded_file)
    img_array = image.img_to_array(img)

    # Apply selected processing method
    if resize_option == "Resize with Padding (Maintain Aspect Ratio)":
        img_array = tf.image.resize_with_pad(img_array, 150, 150)
    else:
        img_array = tf.image.central_crop(img_array, central_fraction=0.8)
        img_array = tf.image.resize(img_array, (150, 150))

    # Preprocess the image
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Display results
    st.write(f"### Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
