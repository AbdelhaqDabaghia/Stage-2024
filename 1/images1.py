import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def trouver_points_intersection(frontieres):
    points_intersection = []

    for i in range(len(frontieres)):
        for j in range(i + 1, len(frontieres)):
            frontiere1 = frontieres[i]
            frontiere2 = frontieres[j]

            if frontiere1[0] == 'y=ax+b' and frontiere2[0] == 'y=ax+b':
                a1, b1 = frontiere1[1], frontiere1[2]
                a2, b2 = frontiere2[1], frontiere2[2]
                
                if a1 != a2:
                    x_intersection = (b2 - b1) / (a1 - a2)
                    y_intersection = a1 * x_intersection + b1
                    if not (np.isnan(x_intersection) or np.isinf(x_intersection) or np.isnan(y_intersection) or np.isinf(y_intersection)):
                        points_intersection.append((x_intersection, y_intersection))

    return points_intersection

def detect_non_linear_boundaries(image_path):
    # Lire l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Détection des contours
    edges = cv2.Canny(blurred, 50, 150)
    
    # Trouver les contours dans l'image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours non linéaires
    non_linear_contours = []
    
    for contour in contours:
        if len(contour) < 5:
            continue  # Ignorer les petits contours qui ne sont pas significatifs
        
        # Approximation polygonale pour vérifier la non-linéarité
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) > 5:  # Seuil pour considérer un contour comme non linéaire
            non_linear_contours.append(contour)
    
    if len(non_linear_contours) < 2:
        raise ValueError("Moins de deux frontières non linéaires détectées.")
    
    # Retourner les deux premiers contours non linéaires détectés
    selected_contour1 = non_linear_contours[0]
    selected_contour2 = non_linear_contours[1]
    
    # Extraire les coordonnées des points d'extrémité des contours sélectionnés
    endpoints_contour1 = [tuple(selected_contour1[0][0]), tuple(selected_contour1[-1][0])]
    endpoints_contour2 = [tuple(selected_contour2[0][0]), tuple(selected_contour2[-1][0])]
    
    return endpoints_contour1, endpoints_contour2

def plot_boundaries(image_path, contour1, contour2):
    # Lire l'image pour l'affichage
    image = cv2.imread(image_path)
    
    # Dessiner les contours sur l'image en rouge
    cv2.drawContours(image, [contour1], -1, (0, 0, 255), 2)
    cv2.drawContours(image, [contour2], -1, (0, 0, 255), 2)
    
    # Afficher l'image avec les contours
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Contours détectés')
    plt.show()

def traiter_images(repertoire):
    fichiers = os.listdir(repertoire)
    donnees = []

    for fichier in fichiers:
        if fichier.endswith('.tif') ou fichier.endswith('.tiff'):
            chemin_image = os.path.join(repertoire, fichier)
            image = cv2.imread(chemin_image)

            try:
                # Détection des frontières non linéaires
                contour1, contour2 = detect_non_linear_boundaries(chemin_image)

                # Affichage des résultats pour chaque image
                print(f"Image: {fichier}")
                print("Contours détectés et tracés.")
                plot_boundaries(chemin_image, contour1, contour2)

                # Dessiner les contours sur l'image en rouge
                cv2.drawContours(image, [contour1], -1, (0, 0, 255), 2)
                cv2.drawContours(image, [contour2], -1, (0, 0, 255), 2)

                # Enregistrer l'image avec les frontières dessinées
                output_path = os.path.join(repertoire, f"{os.path.splitext(fichier)[0]}_avec_frontieres.tif")
                cv2.imwrite(output_path, image)

                points_intersection = trouver_points_intersection([(contour1, (0, 255, 0)), (contour2, (0, 0, 255))])
                y_values = [point[1] for point in points_intersection]

                # Triez les valeurs de y
                y_values.sort()

                # Définissez un seuil de tolérance pour déterminer la proximité des valeurs
                tolerance = 20  # Vous pouvez ajuster ce seuil selon vos besoins

                # Regroupez les valeurs de y proches les unes des autres dans des ensembles
                ensembles = []
                ensemble = [y_values[0]]
                for i in range(1, len(y_values)):
                    if abs(y_values[i] - ensemble[-1]) <= tolerance:
                        ensemble.append(y_values[i])
                    else:
                        ensembles.append(ensemble)
                        ensemble = [y_values[i]]
                ensembles.append(ensemble)

                # Calculez la moyenne de chaque ensemble
                for ensemble in ensembles:
                    moyenne_ensemble = np.mean(ensemble)
                    donnees.append({'Image': fichier, 'Y': moyenne_ensemble})
                    print(f"Moyenne dans un ensemble de valeurs proches pour l'image {fichier}: {moyenne_ensemble}")

            except ValueError as e:
                print(f"Erreur avec l'image {fichier}: {e}")

    df = pd.DataFrame(donnees)
    df.to_excel(os.path.join(repertoire, 'resultats_y.xlsx'), index=False)
    print(f"Les résultats ont été enregistrés dans 'resultats_y.xlsx'.")

# Chemin du répertoire contenant les images TIFF
repertoire_images = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1'
traiter_images(repertoire_images)
