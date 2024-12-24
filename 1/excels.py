import pandas as pd

# Fonction pour extraire les 5 premières colonnes d'un fichier Excel
def extract_first_five_columns(file_path):
    # Lire le fichier Excel
    df = pd.read_excel(file_path)
    # Sélectionner les 5 premières colonnes
    first_five_columns = df.iloc[:, :5]
    return first_five_columns

# Chemins vers les fichiers Excel d'entrée
file1 = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1/29-04-2024.xlsx'
file2 = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1/Alpha_4{2}25_04_2024.xlsx'
file3 = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1/Classeur24_04_2024.xlsx'

# Extraire les 5 premières colonnes de chaque fichier
df1 = extract_first_five_columns(file1)
df2 = extract_first_five_columns(file2)
df3 = extract_first_five_columns(file3)

# Combiner les DataFrames dans un seul
combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# Chemin vers le fichier Excel de sortie
output_file = '/media/dabaghia/UBUNTU 24_0/Mes doc Windows/Stage2024/antoine_a_traiter/12_19_plan_cone_inf/alpha_5/1/fichier_sortie.xlsx'

# Sauvegarder le DataFrame combiné dans un nouveau fichier Excel
combined_df.to_excel(output_file, index=False)

print(f"Les données ont été sauvegardées dans {output_file}")
