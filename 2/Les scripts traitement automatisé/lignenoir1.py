import os
import cv2
import numpy as np
import pandas as pd

def trouver_lignes_interessantes(image_path):
    # Lire l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Vérifier que l'image est chargée correctement
    if image is None:
        print(f"Erreur lors du chargement de l'image {image_path}")
        return None, None, None, None, None, None, None, None
    
    # Initialiser les variables pour trouver les lignes d'intérêt
    min_noirs = float('inf')
    ligne_min_noirs = -1
    max_noirs = -1
    ligne_max_noirs = -1
    meilleure_ligne_blancs = -1
    max_blancs = -1
    x1, x2 = None, None
    
    # Parcourir chaque ligne de l'image
    for y in range(image.shape[0]):
        # Compter le nombre de pixels noirs dans la ligne
        nombre_noirs = np.sum(image[y, :] == 0)
        nombre_blancs = np.sum(image[y, :] == 255)
        
        # Mettre à jour si cette ligne a moins de pixels noirs
        if nombre_noirs < min_noirs:
            min_noirs = nombre_noirs
            ligne_min_noirs = y
            # Trouver les frontières x1 et x2 pour cette ligne
            noirs_indices = np.where(image[y, :] == 0)[0]
            if noirs_indices.size > 0:
                x1, x2 = noirs_indices[0], noirs_indices[-1]
        
        # Mettre à jour si cette ligne a plus de pixels noirs
        if nombre_noirs > max_noirs:
            max_noirs = nombre_noirs
            ligne_max_noirs = y
    
    # Calculer la ligne y' = y + 10
    if ligne_max_noirs != -1 and ligne_max_noirs + 10 < image.shape[0]:
        ligne_y_prime = ligne_max_noirs + 10
        max_blancs = np.sum(image[ligne_y_prime, :] == 255)
        meilleure_ligne_blancs = ligne_y_prime

    return ligne_min_noirs, min_noirs, ligne_max_noirs, max_noirs, meilleure_ligne_blancs, max_blancs, x1, x2

def traiter_images(repertoire_images):
    # Liste pour stocker les résultats
    resultats = []

    # Parcourir tous les fichiers du répertoire
    for filename in os.listdir(repertoire_images):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            # Chemin complet de l'image
            image_path = os.path.join(repertoire_images, filename)
            
            # Trouver les lignes intéressantes
            ligne_min_noirs, min_noirs, ligne_max_noirs, max_noirs, ligne_blanche, max_blancs, x1, x2 = trouver_lignes_interessantes(image_path)
            
            if ligne_min_noirs is not None and ligne_max_noirs is not None and ligne_blanche is not None and x1 is not None and x2 is not None:
                print(f"Image: {filename}")
                print(f"Ligne avec le moins de points noirs: y = {ligne_min_noirs}, Nombre de points noirs: {min_noirs}")
                print(f"Ligne avec le plus de points noirs: y = {ligne_max_noirs}, Nombre de points noirs: {max_noirs}")
                print(f"Ligne à y + 10 avec le plus de points blancs: y' = {ligne_blanche}, Nombre de points blancs: {max_blancs}")
                print(f"Frontières de la ligne avec le moins de points noirs: x1 = {x1}, x2 = {x2}")
                
                # Calculer all_left et all_right
                l = min(x1, x2) - 150
                r = max(x1, x2) + 150
                all_left = l
                all_right = r

                # Ajouter les résultats à la liste
                resultats.append({
                    'image_name_prefix': filename.replace('.tif', ''),
                    'profile_bridge_up': ligne_min_noirs - 70,
                    'profile_bridge_dn': ligne_min_noirs,
                    'profile_cone_up': ligne_min_noirs + 10,
                    'profile_cone_dn': ligne_min_noirs + 30,
                    'profile_all_lf': all_left,
                    'profile_all_r': all_right,
                    'profile_accuracy_bridge_cone': 5
                })

    # Convertir les résultats en DataFrame
    df = pd.DataFrame(resultats)
    
    # Renommer les colonnes
    df = df.rename(columns={
        'bridge_up': 'profile_bridge_up',
        'bridge_dn': 'profile_bridge_dn',
        'cone_up': 'profile_cone_up',
        'cone_dn': 'profile_cone_dn',
        'all_left': 'profile_all_lf',
        'all_right': 'profile_all_r',
        'alpha': 'profile_accuracy_bridge_cone'
    })
    
    # Enregistrer les résultats dans un fichier Excel
    output_path = os.path.join(repertoire_images, 'resultats_bridge.xlsx')
    df.to_excel(output_path, index=False)
    print(f"Les résultats ont été enregistrés dans '{output_path}'.")


# Chemin du répertoire contenant les images TIFF
repertoire_images = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/2'

# Traiter les images dans le répertoire
traiter_images(repertoire_images)
