from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("data/solarwinds-ace/smooth-solarwinds.csv")
DEFAULT_OUTPUT = Path("data/solarwinds-ace/smooth-solarwinds_filled.csv")
VALUE_COLUMNS = ["Speed", "Density", "Bt", "Bz"]


def fill_nulls_with_previous(input_path: Path, output_path: Path) -> int:
    """Remplace les valeurs vides ou null par la dernière valeur connue, en flux."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    last_values: dict[str, str | None] = {column: None for column in VALUE_COLUMNS}

    with input_path.open("r", encoding="utf-8", newline="") as input_handle, output_path.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(input_handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"Fichier CSV vide ou sans en-tête: {input_path}")

        writer = csv.DictWriter(output_handle, fieldnames=reader.fieldnames, delimiter=";")
        writer.writeheader()

        for row in reader:
            for column in VALUE_COLUMNS:
                if column not in row:
                    continue

                value = row[column]
                if value in {"", "null", "NULL"}:
                    previous_value = last_values[column]
                    if previous_value is not None:
                        row[column] = previous_value
                else:
                    last_values[column] = value

            writer.writerow(row)
            total_rows += 1

    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remplace les nulls du fichier SolarWinds par la valeur précédente."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Chemin du CSV source smooth-solarwinds.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Chemin du CSV de sortie rempli.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    row_count = fill_nulls_with_previous(input_path, output_path)

    print(f"CSV source : {input_path}")
    print(f"CSV écrit  : {output_path}")
    print(f"Lignes     : {row_count}")


if __name__ == "__main__":
    main()