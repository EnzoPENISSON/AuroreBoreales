import csv
from pathlib import Path


def generate_files(base_path, start_year, end_year, start_month=1, end_month=12):
    base = Path(base_path)
    return [
        base / str(year) / f"{year}{month:02}.csv"
        for year in range(start_year, end_year + 1)
        for month in range(
            start_month if year == start_year else 1,
            (end_month if year == end_year else 12) + 1,
        )
    ]


X_MIN, X_MAX = 9000, 12000

x_files = generate_files("data/mag-kiruna-compiled", 2007, 2007, 1, 1)

for file in x_files:
    rows = []
    with open(file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames

        for row in reader:
            raw = row["X"]
            try:
                x_value = float(raw)
                if not (X_MIN <= x_value <= X_MAX):
                    row["X"] = ""
            except ValueError:
                row["X"] = ""
                print("Erreur")
            rows.append(row)

    out_file = file.with_stem(file.stem + "_cleaned")
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for i in range(0, len(rows), 10000):
            chunk = rows[i:min(i + 10000, len(rows) + 1)]
            writer.writerows(chunk)

    cleaned = sum(1 for r in rows if r["X"] == "")
    print(f"[OK] {out_file.name}  —  {cleaned}/{len(rows)} valeurs effacées")