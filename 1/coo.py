import cv2
import numpy as np

def detecter_coordonnees(image_noire):
    # Obtenir les dimensions de l'image
    height, width = image_noire.shape
    
    # Initialiser les coordonnées pour le bridge_up
    bridge_up = None
    
    # Parcourir l'image de bas en haut
    for y in range(height-1, -1, -1):
        ligne = image_noire[y, :]
        # Rechercher la première frontière de la forme noire
        if np.any(ligne == 0):
            bridge_up = y
            break
    
    if bridge_up is None:
        raise ValueError("Impossible de détecter le bridge_up.")
    
    return bridge_up

# Test de la fonction avec une image noire (0 signifie noir)
image_noire = np.zeros((10, 10), dtype=np.uint8)
image_noire[3:7, 3:7] = 255  # Ajouter une forme noire au centre
bridge_up = detecter_coordonnees(image_noire)
print("Bridge Up:", bridge_up)
