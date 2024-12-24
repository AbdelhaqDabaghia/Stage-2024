import pandas as pd

# Chemin vers le fichier Excel à traiter
file_path = r"C:\Users\adabag01\Documents\Stage2024\antoine_a_traiter\12_19_plan_cone_inf\alpha_5\Classeur2.xlsx"

# Lire le fichier Excel
df = pd.read_excel(file_path)

# Remplacer les points par des virgules dans toutes les colonnes numériques
for col in df.columns:
    if df[col].dtype == 'float64':  # Vérifier si la colonne contient des nombres décimaux
        df[col] = df[col].astype(str).str.replace('.', ',')

# Enregistrer les modifications dans un nouveau fichier Excel
new_file_path = file_path.replace('.xlsx', '_corrected.xlsx')
df.to_excel(new_file_path, index=False)

print(f"Le fichier Excel corrigé a été enregistré sous '{new_file_path}'.")
