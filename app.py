import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('Models/keras_model.h5')

# Load the labels
with open('Models/labels.txt', 'r') as f:
    labels = f.read().splitlines()


def main():
    st.title('Cataract Eye Problem Detection App')
    st.write('Upload an image of an eye to detect if it has a cataract.')

    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the model
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, (224, 224))
        img_array = tf.expand_dims(img_array, 0)

        # Make predictions using the model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        # Define the classes and corresponding messages
        classes = ['Cataract', 'Normal']
        messages = ["You might have cataract. Please consult an eye specialist.", "Your eyes are normal."]

        # Display the prediction message
        st.write('Prediction:')
        st.write(f'Class: {classes[predicted_class]}')
        st.write(messages[predicted_class])
        st.write('Please note that these results are based on the AI model(Not a real eye specialist) and therefore may not be 100%. accurate.')

if __name__ == '__main__':
    main()
