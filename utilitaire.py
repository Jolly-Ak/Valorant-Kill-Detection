


import pathlib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import tensorflow as tf
from tensorflow.keras import layers
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#load the model
model = tf.keras.models.load_model('model_final_v2.h5')
img_height = 220

img_width = 220
# Get the class names
class_names = ['kill', 'notkill']

predict_dir = pathlib.Path('predict')
for image_path in predict_dir.glob('*.jpg'):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the predicted class based on the highest probability
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = class_names[predicted_class]

    # Rename the image based on the predicted class
    new_image_name = f"{class_name}_{image_path.name}"
    new_image_path = image_path.parent / new_image_name
    image_path.rename(new_image_path)
    #afficher les prediction en % 
    print(f"Prediction: {prediction}, Predicted class: {predicted_class}")  # Affichage pour vérifier
    print(f"Image {image_path} renamed to {new_image_path}")
    print(f"Image {image_path} renamed to {new_image_path}")