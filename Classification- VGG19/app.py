import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from PIL import Image
from PIL import ImageOps
# @st.cache(allow_output_mutation=True)
def loadmodel():
  model = tf.keras.models.load_model('my_model2.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=loadmodel()

def main():
    # Set page title and layout

    st.title("Cat and Dog Classification")

    # Create a sidebar for additional controls
    st.sidebar.title("Options")

    # Upload image
    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display uploaded image
    if uploaded_image is not None:
        st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Save image in a variable
    if st.sidebar.button("Predict Now"):
        if uploaded_image is not None:
            # Perform any processing on the image here
            # For example, you can save it to a variable
            st.image(uploaded_image, use_column_width=True)
            image = Image.open(uploaded_image)
            image_array = np.array(image.resize((150, 150)))
            image_array = np.reshape(image_array, (1, 150, 150, 3))
            data = ['Cat', 'Dog']
            prediction = model.predict(image_array)
            prediction = int(prediction)
            # Perform your prediction or further processing here
            # For example, you can pass the image_data to a pre-trained model for prediction

            # Display the result
            st.success("The Prediction is : "+ data[prediction])

            # Display a success message
            st.sidebar.success("Image saved successfully!")
        else:
            # Display an error message if no image is uploaded
            st.sidebar.error("Please upload an image first.")

if __name__ == "__main__":
    main()
