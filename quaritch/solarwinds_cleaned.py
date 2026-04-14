from pathlib import Path
import pandas as pd

NULL_TOKEN = "null"

RULES = {
    "Speed":   lambda x: 0 < x < 1500,
    "Density": lambda x: 0 < x < 200,
    "Bt":      lambda x: 0 < x < 100,
    "Bz":      lambda x: -50 < x < 50,
}

EXPECTED_COLUMNS = {"Speed", "Density", "Bt", "Bz"}


def normalize_cell(value: object) -> str:
    """
    Normalise une cellule :
    - NaN / vide / blanc -> null
    - trim des espaces
    - remplace la virgule décimale par un point
    """
    if pd.isna(value):
        return NULL_TOKEN

    text = str(value).strip()

    if text == "":
        return NULL_TOKEN

    text = text.replace(",", ".")
    return text


def clean_value(value: str, rule, stats: dict, col_name: str) -> str:
    """
    Nettoie une valeur d'une colonne métier et met à jour les stats.
    """
    stats["columns"][col_name]["total_values"] += 1

    if value == NULL_TOKEN:
        stats["columns"][col_name]["missing_count"] += 1
        return NULL_TOKEN

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        stats["columns"][col_name]["invalid_format_count"] += 1
        return NULL_TOKEN

    if not rule(numeric_value):
        stats["columns"][col_name]["out_of_range_count"] += 1
        return NULL_TOKEN

    stats["columns"][col_name]["valid_count"] += 1
    return value


def initialize_file_stats(df: pd.DataFrame) -> dict:
    """
    Initialise les statistiques d'un fichier.
    """
    return {
        "global": {
            "row_count": len(df),
            "column_count": len(df.columns),
            "total_cells": len(df) * len(df.columns),
            "null_cells": 0,
        },
        "columns": {
            col: {
                "total_values": 0,
                "missing_count": 0,
                "invalid_format_count": 0,
                "out_of_range_count": 0,
                "valid_count": 0,
            }
            for col in RULES.keys()
        }
    }


def initialize_global_stats() -> dict:
    """
    Initialise les statistiques globales sur tous les fichiers.
    """
    return {
        "files": 0,
        "rows": 0,
        "cells": 0,
        "null_cells": 0,
        "columns": {
            col: {
                "total_values": 0,
                "missing_count": 0,
                "invalid_format_count": 0,
                "out_of_range_count": 0,
                "valid_count": 0,
            }
            for col in RULES.keys()
        }
    }


def merge_stats(global_stats: dict, file_stats: dict) -> None:
    """
    Fusionne les stats d'un fichier dans les stats globales.
    """
    global_stats["files"] += 1
    global_stats["rows"] += file_stats["global"]["row_count"]
    global_stats["cells"] += file_stats["global"]["total_cells"]
    global_stats["null_cells"] += file_stats["global"]["null_cells"]

    for col in RULES.keys():
        for key in global_stats["columns"][col]:
            global_stats["columns"][col][key] += file_stats["columns"][col][key]


def print_file_stats(stats: dict) -> None:
    """
    Affiche les statistiques d'un fichier.
    """
    print("\n=== STATISTIQUES DE NETTOYAGE ===")
    print(f"Lignes                    : {stats['global']['row_count']}")
    print(f"Colonnes                  : {stats['global']['column_count']}")
    print(f"Cellules totales          : {stats['global']['total_cells']}")
    print(f"Cellules à null           : {stats['global']['null_cells']}")

    total_cells = stats["global"]["total_cells"]
    if total_cells > 0:
        pct_null = (stats["global"]["null_cells"] / total_cells) * 100
        print(f"% cellules à null         : {pct_null:.4f}%")

    print("\n--- Détail par colonne métier ---")
    for col, col_stats in stats["columns"].items():
        total = col_stats["total_values"]
        if total == 0:
            print(f"{col:<8} : colonne absente ou non traitée")
            continue

        print(f"{col:<8} :")
        print(f"  Total analysé           : {col_stats['total_values']}")
        print(f"  Valides                 : {col_stats['valid_count']}")
        print(f"  Vides -> null           : {col_stats['missing_count']}")
        print(f"  Format invalide -> null : {col_stats['invalid_format_count']}")
        print(f"  Hors intervalle -> null : {col_stats['out_of_range_count']}")


def print_global_stats(global_stats: dict) -> None:
    """
    Affiche les statistiques globales sur tous les fichiers.
    """
    print(f"\n{'#' * 80}")
    print("STATISTIQUES GLOBALES")
    print(f"Fichiers traités         : {global_stats['files']}")
    print(f"Lignes totales           : {global_stats['rows']}")
    print(f"Cellules totales         : {global_stats['cells']}")
    print(f"Cellules à null          : {global_stats['null_cells']}")

    if global_stats["cells"] > 0:
        pct_null = (global_stats["null_cells"] / global_stats["cells"]) * 100
        print(f"% global de null         : {pct_null:.4f}%")

    print("\n--- Détail global par colonne ---")
    for col, stats in global_stats["columns"].items():
        print(f"\n{col} :")
        print(f"  Total analysé           : {stats['total_values']}")
        print(f"  Valides                 : {stats['valid_count']}")
        print(f"  Vides -> null           : {stats['missing_count']}")
        print(f"  Format invalide -> null : {stats['invalid_format_count']}")
        print(f"  Hors intervalle -> null : {stats['out_of_range_count']}")

    print(f"{'#' * 80}")


def clean_one_csv(csv_path: Path, convert_for_ml: bool = False) -> tuple[bool, dict | None]:
    """
    Nettoie un seul fichier CSV.
    Retourne (True, stats) si le traitement s'est bien passé, sinon (False, None).
    """
    if not csv_path.exists():
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return False, None

    if csv_path.suffix.lower() != ".csv":
        print(f"[ERREUR] Le fichier n'est pas un CSV : {csv_path}")
        return False, None

    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        raw_input_line_count = sum(1 for _ in f)

    try:
        df = pd.read_csv(
            csv_path,
            sep=";",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            engine="python"
        )
    except Exception as e:
        print(f"[ERREUR LECTURE] {csv_path} -> {e}")
        return False, None

    input_row_count = len(df)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        print(f"[WARNING] Colonnes manquantes dans {csv_path} : {sorted(missing_cols)}")

    extra_cols = set(df.columns) - (EXPECTED_COLUMNS | {"Date"})
    if extra_cols:
        print(f"[INFO] Colonnes supplémentaires détectées dans {csv_path.name} : {sorted(extra_cols)}")

    stats = initialize_file_stats(df)

    # Normalisation globale
    df = df.apply(lambda col: col.map(normalize_cell))

    # Comptage global des null après normalisation
    stats["global"]["null_cells"] = int((df == NULL_TOKEN).sum().sum())

    # Nettoyage des colonnes métier
    for col, rule in RULES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda v: clean_value(v, rule, stats, col))

    # Recalcul du nombre global de null après nettoyage métier
    stats["global"]["null_cells"] = int((df == NULL_TOKEN).sum().sum())

    if convert_for_ml:
        for col in RULES.keys():
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].replace(NULL_TOKEN, pd.NA),
                    errors="coerce"
                )

    output_path = csv_path.with_name(f"{csv_path.stem}_clean.csv")

    try:
        df.to_csv(output_path, sep=";", index=False, lineterminator="\n")
    except Exception as e:
        print(f"[ERREUR ECRITURE] {output_path} -> {e}")
        return False, None

    with open(output_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        raw_output_line_count = sum(1 for _ in f)

    try:
        df_check = pd.read_csv(
            output_path,
            sep=";",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            engine="python"
        )
        output_row_count = len(df_check)
    except Exception as e:
        print(f"[ERREUR RELECTURE] {output_path} -> {e}")
        return False, None

    print("\n=== VERIFICATION ===")
    print(f"Fichier source        : {csv_path}")
    print(f"Fichier nettoyé       : {output_path}")
    print(f"Lignes physiques src  : {raw_input_line_count}")
    print(f"Lignes physiques out  : {raw_output_line_count}")
    print(f"Lignes de données src : {input_row_count}")
    print(f"Lignes de données out : {output_row_count}")

    if input_row_count != output_row_count:
        print("[ERREUR] Le nombre de lignes de données a changé.")
        return False, None

    print("[OK] Le nombre de lignes de données est identique.")
    print_file_stats(stats)
    return True, stats


def process_root_folder(
    root_folder: str,
    global_stats: dict,
    convert_for_ml: bool = False
) -> tuple[int, int]:
    """
    Parcourt récursivement un dossier racine et traite tous les CSV.
    Ignore les fichiers déjà suffixés par _clean.csv.
    Retourne (nb_succes, nb_echecs).
    """
    root = Path(root_folder)

    if not root.exists():
        print(f"[ERREUR] Dossier introuvable : {root}")
        return 0, 1

    if not root.is_dir():
        print(f"[ERREUR] Ce n'est pas un dossier : {root}")
        return 0, 1

    csv_files = sorted(
        [p for p in root.rglob("*.csv") if not p.stem.endswith("_clean")],
        key=lambda p: str(p)
    )

    print(f"\n=== DOSSIER RACINE : {root} ===")
    print(f"Nombre de CSV à traiter : {len(csv_files)}")

    success_count = 0
    failure_count = 0

    for i, csv_file in enumerate(csv_files, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(csv_files)}] Traitement de : {csv_file}")

        ok, file_stats = clean_one_csv(csv_file, convert_for_ml=convert_for_ml)

        if ok:
            success_count += 1
            merge_stats(global_stats, file_stats)
        else:
            failure_count += 1

    print(f"\n=== FIN DOSSIER : {root} ===")
    print(f"Succès : {success_count}")
    print(f"Échecs : {failure_count}")

    return success_count, failure_count


def main(root1: str, root2: str, convert_for_ml: bool = False) -> None:
    total_success = 0
    total_failure = 0
    global_stats = initialize_global_stats()

    success1, failure1 = process_root_folder(
        root1,
        global_stats,
        convert_for_ml=convert_for_ml
    )
    total_success += success1
    total_failure += failure1

    success2, failure2 = process_root_folder(
        root2,
        global_stats,
        convert_for_ml=convert_for_ml
    )
    total_success += success2
    total_failure += failure2

    print(f"\n{'#' * 80}")
    print("BILAN GLOBAL")
    print(f"Fichiers traités avec succès : {total_success}")
    print(f"Fichiers en échec            : {total_failure}")
    print(f"{'#' * 80}")

    print_global_stats(global_stats)


if __name__ == "__main__":
    dossier_racine_1 = "data/solarwinds-ace-compiled"
    dossier_racine_2 = "data/solarwinds-dscovr-compiled"

    # False = garde "null" comme texte dans les CSV
    # True  = convertit les colonnes métier en numérique avec NaN
    main(dossier_racine_1, dossier_racine_2, convert_for_ml=False)