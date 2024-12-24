import os
import cv2
import numpy as np

def is_symmetric(contour1, contour2, axis_x):
    # Symétrie approximative en vérifiant si la distance de chaque point au centre de symétrie est similaire
    for point1, point2 in zip(contour1, contour2):
        if not np.isclose(point1[0][0], 2 * axis_x - point2[0][0], atol=5):
            return False
    return True

def colorize_boundaries(image_dir):
    # Liste des couleurs pour différentes frontières
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            filepath = os.path.join(image_dir, filename)
            print(f"Processing file: {filepath}")
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Error loading image: {filepath}")
                continue

            height, width = image.shape
            mid_x = width // 2

            # Diviser l'image en morceaux horizontaux
            segments = np.array_split(image, height // 100)  # Changez 100 selon la hauteur souhaitée des segments

            for idx, segment in enumerate(segments):
                # Seuil pour binariser l'image (noir et blanc)
                _, binary_image = cv2.threshold(segment, 127, 255, cv2.THRESH_BINARY_INV)

                # Trouver les contours
                contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Convertir le segment en BGR pour la colorisation
                color_segment = cv2.cvtColor(segment, cv2.COLOR_GRAY2BGR)

                # Vérifier les symétries
                for i, contour1 in enumerate(contours):
                    for j, contour2 in enumerate(contours):
                        if i >= j:
                            continue
                        if is_symmetric(contour1, contour2, mid_x):
                            color = colors[2]  # Couleur pour les contours symétriques
                            cv2.drawContours(color_segment, [contour1], -1, color, 2)
                            cv2.drawContours(color_segment, [contour2], -1, color, 2)

                            # Calculer et imprimer les valeurs maximale et minimale de y pour les contours symétriques
                            y_values1 = contour1[:, 0, 1]
                            y_values2 = contour2[:, 0, 1]
                            max_y1 = np.max(y_values1)
                            min_y1 = np.min(y_values1)
                            max_y2 = np.max(y_values2)
                            min_y2 = np.min(y_values2)
                            print(f"Max y for contour1 in segment {idx}: {max_y1}")
                            print(f"Min y for contour1 in segment {idx}: {min_y1}")
                            print(f"Max y for contour2 in segment {idx}: {max_y2}")
                            print(f"Min y for contour2 in segment {idx}: {min_y2}")

                        else:
                            epsilon1 = 0.01 * cv2.arcLength(contour1, True)
                            approx1 = cv2.approxPolyDP(contour1, epsilon1, True)
                            epsilon2 = 0.01 * cv2.arcLength(contour2, True)
                            approx2 = cv2.approxPolyDP(contour2, epsilon2, True)

                            color1 = colors[0] if len(approx1) == 4 else colors[1]
                            color2 = colors[0] if len(approx2) == 4 else colors[1]
                            cv2.drawContours(color_segment, [contour1], -1, color1, 2)
                            cv2.drawContours(color_segment, [contour2], -1, color2, 2)

                # Sauvegarder le segment colorisé
                segment_output_path = os.path.join(image_dir, f'colorized_segment_{idx}_{filename}')
                cv2.imwrite(segment_output_path, color_segment)

if __name__ == "__main__":
    image_directory = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1'
    colorize_boundaries(image_directory)
