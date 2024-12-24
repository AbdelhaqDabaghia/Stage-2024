import os
import cv2
import numpy as np

def trouver_ligne_min_points_noirs(image_path):
    # Lire l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Vérifier que l'image est chargée correctement
    if image is None:
        print(f"Erreur lors du chargement de l'image {image_path}")
        return None
    
    # Initialiser les variables pour trouver la ligne avec le moins de points noirs
    min_noirs = float('inf')
    meilleure_ligne = -1
    
    # Parcourir chaque ligne de l'image
    for y in range(image.shape[0]):
        # Compter le nombre de pixels noirs dans la ligne
        nombre_noirs = np.sum(image[y, :] == 0)
        
        # Mettre à jour si cette ligne a moins de pixels noirs
        if nombre_noirs < min_noirs:
            min_noirs = nombre_noirs
            meilleure_ligne = y
    
    return meilleure_ligne, min_noirs

def traiter_images(repertoire_images):
    # Parcourir tous les fichiers du répertoire
    for filename in os.listdir(repertoire_images):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # Chemin complet de l'image
            image_path = os.path.join(repertoire_images, filename)
            
            # Trouver la ligne avec le moins de points noirs
            meilleure_ligne, min_noirs = trouver_ligne_min_points_noirs(image_path)
            
            if meilleure_ligne is not None:
                print(f"Image: {filename}")
                print(f"Meilleure ligne: y = {meilleure_ligne}, Nombre de points noirs: {min_noirs}")
                # Lire l'image en couleur pour dessiner la ligne
                image_couleur = cv2.imread(image_path)
                # Dessiner la ligne horizontale en bleu (BGR)
                cv2.line(image_couleur, (0, meilleure_ligne), (image_couleur.shape[1], meilleure_ligne), (255, 0, 0), 1)
                # Afficher l'image avec la ligne
                plt.figure(figsize=(8, 8))
                plt.imshow(cv2.cvtColor(image_couleur, cv2.COLOR_BGR2RGB))
                plt.title(f"Ligne horizontale y={meilleure_ligne} pour {os.path.basename(image_path)}")
                plt.axis('off')
                plt.show()

# Chemin du répertoire contenant les images TIFF
repertoire_images = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1'

# Traiter les images dans le répertoire
traiter_images(repertoire_images)
