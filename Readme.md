# Valorant Kill Detection using Deep Learning

Ce projet utilise un modèle de deep learning pour détecter automatiquement les moments de "kill" dans des vidéos de **Valorant**. Le modèle est formé à partir d'un jeu d'images étiquetées représentant des moments de kill et non-kill. Il utilise un réseau de neurones convolutif (CNN) pour classer les images.

## Fonctionnalités
- **Prédiction**: Le modèle peut prédire si une image donnée correspond à un moment de kill ou non dans une vidéo de **Valorant**.
- **Prétraitement des images**: Redimensionnement et ajout de padding pour les images avant de les soumettre au modèle.
- **Prédiction en lot**: Le modèle peut renommer automatiquement toutes les images dans un répertoire en fonction de la classe prédite (kill / not kill).
<div align="center">

*Figure du d'activation de la 2éme couche de convolution*  
![alt text](Figure/Figure_1.png)

</div>

## Prérequis

Assurez-vous que vous avez installé les bibliothèques suivantes:
- `tensorflow` : pour créer et entraîner le modèle.
- `keras` : utilisé avec TensorFlow pour faciliter la création du modèle.
- `opencv-python` : pour le traitement d'images.
- `numpy` : pour la manipulation des matrices et tableaux.
- `matplotlib` : pour afficher des graphiques et images.
- `pathlib` : pour la gestion des chemins de fichiers.

### Installation
1. Clonez le repository:
    ```bash
    git clone https://github.com/yourusername/valorant-kill-detection.git
    cd valorant-kill-detection
    ```

2. Créez un environnement virtuel et installez les dépendances:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Pour Linux/MacOS
    venv\Scripts\activate     # Pour Windows
    pip install -r requirements.txt
    ```

3. Assurez-vous d'avoir un répertoire `dataset` et `val_data` contenant des images étiquetées comme "kill" et "not_kill".

## Structure du Projet

- `dataset/`: Contient les images d'entraînement avec des sous-dossiers `kill` et `not_kill`.
- `val_data/`: Contient des images de validation pour tester le modèle.
- `predict/`: Contient des images non étiquetées que vous souhaitez prédire et renommer.
- `model_final.h5`: Le modèle entraîné sauvegardé après l'entraînement.
- `train_model.py`: Script principal pour entraîner le modèle et effectuer les prédictions.

## Entraînement du modèle

Le modèle est un réseau de neurones convolutifs (CNN) qui suit les étapes suivantes:

1. **Prétraitement des images**: Redimensionnement à une taille uniforme et ajout de padding si nécessaire.
2. **Création du modèle**: Le modèle consiste en plusieurs couches convolutives et de pooling pour extraire les caractéristiques importantes des images.
3. **Compilation et entraînement**: Le modèle est compilé avec l'optimiseur Adam et la fonction de perte `SparseCategoricalCrossentropy`, puis il est entraîné pendant 15 époques sur le jeu de données d'entraînement.
4. **Évaluation**: Le modèle est évalué sur le jeu de données de validation pour vérifier ses performances.

Pour entraîner le modèle, lancez simplement:

```bash
python train_model.py
