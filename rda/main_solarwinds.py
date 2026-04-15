import csv
from pathlib import Path

def generate_files(base_path, start_year, end_year, start_month=1, end_month=12):
    files = []

    for year in range(start_year, end_year + 1):
        for month in range(start_month if start_year == year else 1, end_month + 1 if end_year == year else 13):
            files.append(f"{base_path}/{year}/{year}{month:02}_clean.csv")

    return files

kp_files = ["data/kp-compiled/kp_cleaned.csv"]
x_files = generate_files("data/mag-kiruna-compiled", 2007, 2024, 1, 6)

# Ace
solar_wind_files_ace = generate_files(
    "data/solarwinds-ace-compiled",
    2007,
    2016,
    1,
    7
)

# Discover
solar_wind_files_dscovr = generate_files(
    "data/solarwinds-dscovr-compiled",
    2016,
    2024,
    7,
    6
)

def run():
    out_file = "data/solarwinds-ace-compiled/smooth.csv"
    with open(out_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Speed", "Density", "Bt", "Bz"], delimiter=";")
        writer.writeheader()

    for file in solar_wind_files_ace:
        print(file)

        rows = []
        with open(file, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            fieldnames = reader.fieldnames

            rows = []

            for row in reader:
                rows.append(row)
  
        with open(out_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

            for i in range(0, len(rows), 10000):
                chunk = rows[i:min(i + 10000, len(rows) + 1)]
                writer.writerows(chunk)

    for file in solar_wind_files_dscovr:
        print(file)

        rows = []
        with open(file, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            fieldnames = reader.fieldnames

            index = 0
            speed_values = []
            density_values = []
            bt_values = []
            bz_values = []
            
            for row in reader:
                if row["Speed"] != "" and row["Speed"] != "null":
                    speed_values.append(float(row["Speed"]))
                if row["Density"] != "" and row["Density"] != "null":
                    density_values.append(float(row["Density"]))
                if row["Bt"] != "" and row["Bt"] != "null":
                    bt_values.append(float(row["Bt"]))
                if row["Bz"] != "" and row["Bz"] != "null":
                    bz_values.append(float(row["Bz"]))

                if index % 15 == 14:
                    newRow = row.copy()
                    newRow["Speed"] = smooth(speed_values)
                    newRow["Density"] = smooth(density_values)
                    newRow["Bt"] = smooth(bt_values)
                    newRow["Bz"] = smooth(bz_values)
                    rows.append(newRow)
                    speed_values = []
                    density_values = []
                    bt_values = []
                    bz_values = []

                index += 1
  
        with open(out_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

            for i in range(0, len(rows), 10000):
                chunk = rows[i:min(i + 10000, len(rows) + 1)]
                writer.writerows(chunk)

def smooth(inputs):
    avg = 0

    for x in inputs:
        avg += x

    if len(inputs) != 0:
        avg /= len(inputs)

    return round(avg, 2)

if __name__ == '__main__':
    run()