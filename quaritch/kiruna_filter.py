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


x_files = generate_files("data/mag-kiruna-compiled", 2007, 2007, 1, 6)

values = set()
for file in x_files:
    with open(file, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")   # row["X"] fonctionne
        for row in reader:
            values.add(int(float(row["X"]) / 10) * 10)  # 10505.7 → 10500

print(sorted(values))