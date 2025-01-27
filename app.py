import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model("xray_classification_model.keras")

# Class names
class_names = ["Healthy", "Pneumonia"]

st.title("X-ray Image Classifier")
st.write("Upload an X-ray image to classify if it's healthy or shows pneumonia.")

# Upload an image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    
    # Since the model outputs a single value (sigmoid), calculate probabilities
    probability_pneumonia = prediction[0][0]  # Probability of "Pneumonia"
    probability_healthy = 1 - probability_pneumonia  # Probability of "Healthy"
    
    # Determine the predicted class
    if probability_pneumonia > 0.5:
        predicted_class = class_names[1]
        confidence = probability_pneumonia * 100
    else:
        predicted_class = class_names[0]
        confidence = probability_healthy * 100

    # Display results
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.markdown(
        f"<h2 style='text-align: center; color: green;'>Predicted: {predicted_class} ({confidence:.2f}%)</h2>",
        unsafe_allow_html=True
    )
    st.write(
        f"Confidence: {class_names[0]}: {probability_healthy * 100:.2f}%, "
        f"{class_names[1]}: {probability_pneumonia * 100:.2f}%"
    )
