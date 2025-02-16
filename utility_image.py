import cv2

import numpy as np

def detect_edges(path):
    image = cv2.imread(path)
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Utiliser un détecteur de contours pour trouver les objets dans l'image
    edges = cv2.Canny(gray_image, 100, 200)

    
    cv2.imshow("Edges", edges)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kernel_horizontal(path):
    image = cv2.imread(path)
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Créer un noyau horizontal
    kernel = np.array([[-1, -1, -1],
                       [2, 2, 2],
                       [-1, -1, -1]])

    # Appliquer le noyau sur l'image
    filtered_image = cv2.filter2D(gray_image, -1, kernel)

    cv2.imshow("Horizontal Kernel", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kernel_vertical(path):
    image = cv2.imread(path)
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Créer un noyau vertical
    kernel = np.array([[-1, 2, -1],
                       [-1, 2, -1],
                       [-1, 2, -1]])

    # Appliquer le noyau sur l'image
    filtered_image = cv2.filter2D(gray_image, -1, kernel)

    cv2.imshow("Vertical Kernel", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

