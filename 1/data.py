import os
import cv2
import numpy as np
import pandas as pd

def get_segment_y_range(segment_image, original_image):
    """
    Trouve l'intervalle de y de l'image du segment dans l'image originale.
    """
    segment_height = segment_image.shape[0]
    height, width = original_image.shape

    for y in range(height - segment_height + 1):
        for x in range(width - segment_image.shape[1] + 1):
            roi = original_image[y:y + segment_height, x:x + segment_image.shape[1]]
            if np.array_equal(segment_image, roi):
                return y, y + segment_height - 1
    return None, None

def process_segments(image_dir, segment_prefix='colorized_segment_23_'):
    segments = [f for f in os.listdir(image_dir) if f.startswith(segment_prefix) and (f.endswith('.tif') or f.endswith('.tiff'))]
    intervals = []

    for segment in segments:
        image_name = segment.split('_')[-1]  # extrait le nom de l'image originale
        image_path = os.path.join(image_dir, image_name)

        # Lire l'image initiale
        initial_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if initial_image is None:
            print(f"Erreur lors de la lecture de l'image initiale : {image_path}")
            continue

        # Lire le segment
        segment_path = os.path.join(image_dir, segment)
        segment_image = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
        if segment_image is None:
            print(f"Erreur lors de la lecture du segment : {segment}")
            continue

        # Trouver l'intervalle de y du segment dans l'image initiale
        y_start, y_end = get_segment_y_range(segment_image, initial_image)
        if y_start is None or y_end is None:
            print(f"Impossible de trouver le segment {segment} dans l'image initiale {image_name}")
            continue

        intervals.append({
            'Image Initiale': image_name,
            'Segment': segment,
            'y_start': y_start,
            'y_end': y_end
        })

    # Enregistrer les résultats dans un fichier Excel
    df = pd.DataFrame(intervals)
    df.to_excel(os.path.join(image_dir, 'y_intervals_segments_23.xlsx'), index=False)
    print(f"Les intervalles de y ont été enregistrés dans 'y_intervals_segments_23.xlsx'.")

if __name__ == "__main__":
    image_directory = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1'
    process_segments(image_directory)
