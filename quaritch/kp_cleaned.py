"""
Filtre les lignes du fichier kp.csv dont la valeur Kp est < 1 ou > 9.
"""
import pandas as pd
from pathlib import Path

INPUT_FILE  = Path(r"data/kp-compiled/kp.csv")
OUTPUT_FILE = Path(r"data/kp-compiled/kp_cleaned.csv")

# Chargement
df = pd.read_csv(INPUT_FILE, sep=";")
total_avant = len(df)

# Conversion numérique (gère les valeurs non numériques éventuelles)
df["Kp"] = pd.to_numeric(df["Kp"], errors="coerce")

# Suppression des lignes hors plage : Kp < 1 ou Kp > 9
mask_a_supprimer = (df["Kp"] < 1) | (df["Kp"] > 9)
n_supprimees = mask_a_supprimer.sum()

df_filtre = df[~mask_a_supprimer].reset_index(drop=True)

# Export

df_filtre.to_csv(OUTPUT_FILE, sep=";", index=False)

# Rapport
print(f"Lignes avant filtrage  : {total_avant:>8,}")
print(f"Lignes supprimées      : {n_supprimees:>8,}  (Kp < 1 ou Kp > 9)")
print(f"Lignes après filtrage  : {len(df_filtre):>8,}")
print(f"Fichier sauvegardé     : {OUTPUT_FILE}")