import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def visualize_filters(model_path, image_path):
    # Charger le modèle
    model = load_model(model_path)

    # Forcer l'initialisation du modèle avec un tenseur factice
    img_height, img_width = 220, 220  # Taille utilisée lors de l'entraînement
    dummy_input = tf.random.normal((1, img_height, img_width, 3))  # Image factice
    model(dummy_input)  # Appel pour initialiser le modèle

    # Charger et prétraiter l'image
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter la dimension batch
    img_array /= 255.0  # Normalisation
    
    # Extraire les couches de convolution du modèle
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    
    if not conv_layers:
        print("Le modèle ne contient pas de couches convolutionnelles.")
        return
    
    # Vérifier les couches récupérées
    print("Convolutional layers found:", [layer.name for layer in conv_layers])

    # Utiliser directement l'entrée existante du modèle après initialisation
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in conv_layers])
    
    # Obtenir les activations intermédiaires
    intermediate_outputs = intermediate_model.predict(img_array)
        
    # Afficher les sorties des couches convolutives
    for i, feature_map in enumerate(intermediate_outputs):
        num_filters = feature_map.shape[-1]
        fig, axes = plt.subplots(1, min(num_filters, 5), figsize=(15, 5))  # Limite l'affichage à 5 filtres
        fig.suptitle(f"Activation de la couche {conv_layers[i].name}")
        
        for j in range(min(num_filters, 5)):
            ax = axes[j] if num_filters > 1 else axes
            ax.imshow(feature_map[0, :, :, j], cmap='viridis')
            ax.axis('off')
        
        plt.show()


# Exécuter la visualisation
model_path = 'model_final.h5'
image_path = 'predict/kill_video_35_frame_12s.jpg'
visualize_filters(model_path, image_path)
