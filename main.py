import cv2
import numpy as np
from utility_image import detect_edges, kernel_horizontal, kernel_vertical
from utility_dir import image_count
from tensorflow import keras
# TODO : tchek if its working with keras of tensorflow
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import pathlib
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import datetime


# Disable oneDNN custom operations to avoid numerical differences
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
data_dir = pathlib.Path('./dataset')
val_dir = pathlib.Path('./val_data')
image_kill = "1739704602523.jpeg"
kill_path = "dataset/kill"
not_kill_path = "dataset/not_kill"


img_height = 220
img_width = 220
batch_size = 4

def preprocess_image(image):
    # Redimensionner l'image en maintenant le ratio d'origine et ajouter du padding
    image = tf.image.resize_with_crop_or_pad(image, target_height=img_height, target_width=img_width)
    return image

# Appliquer le prétraitement lors de la création des datasets
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),  # Ne modifie pas directement l'image, mais utilise la fonction de prétraitement
    batch_size=batch_size,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    validation_split=0.2,
    seed=42,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

# Appliquer la fonction de prétraitement sur chaque image dans le dataset
class_names = train_data.class_names

num_classes = len(class_names)
print(class_names)
for image_batch, labels_batch in train_data:
    print(np.bincount(labels_batch.numpy()))

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


model = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(128,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),  # from_logits=True si ta couche de sortie n'utilise pas softmax
              metrics=['accuracy'])

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True)

model.fit( 
    train_data,
  validation_data=val_data,
  epochs=15,
  callbacks=[tensorboard_callback]
)

model.save('model_final.h5')
model.summary()


# redict image_kill 

img = load_img(image_kill, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour batch size

# Faire la prédiction
prediction = model.predict(img_array)

# Obtenir la classe prédite en fonction de la probabilité la plus élevée
predicted_class = np.argmax(prediction, axis=1)  # Indice de la classe prédite
print(f"Image {image_kill} est de la classe {predicted_class}")
# Obtenir la classe prédite en fonction de la probabilité la plus élevée
predicted_class = np.argmax(prediction, axis=1)  # Indice de la classe prédite
print(f"Prediction: {prediction}, Predicted class: {predicted_class}")  # Affichage pour vérifier



# Load and preprocess a new image for prediction
# Classify and rename all images in the 'predict' directory
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
    print(f"Image {image_path} renamed to {new_image_path}")
    print(f"Image {image_path} renamed to {new_image_path}")