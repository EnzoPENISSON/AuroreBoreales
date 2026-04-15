import csv

rows = []

out_file = "data/kp-compiled/smooth.csv"
with open(out_file, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["Date", "Kp"], delimiter=";")
    writer.writeheader()

with open("data/kp-compiled/kp.csv") as f:
    reader = csv.DictReader(f, delimiter=";")
    fieldnames = reader.fieldnames

    for row in reader:
        rows.append(row)
        if int(row["Date"]) < 20160726000000:
            for i in range(1, 180):
                dupRow = row.copy()
                dupRow["Date"] = int(dupRow["Date"])
                dupRow["Date"] += (i % 60) * 100
                dupRow["Date"] += (i // 60) * 10000
                dupRow["Date"] = str(dupRow["Date"])
                rows.append(dupRow)
        else:
            for i in range(1, 720):
                dupRow = row.copy()
                dupRow["Date"] = int(dupRow["Date"])
                dupRow["Date"] += (i % 4) * 15
                dupRow["Date"] += ((i // 4) - 60 * (i // 240)) * 100
                dupRow["Date"] += (i // 240) * 10000
                dupRow["Date"] = str(dupRow["Date"])
                rows.append(dupRow)

with open(out_file, "a") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

    for i in range(0, len(rows), 10000):
        chunk = rows[i:min(i + 10000, len(rows) + 1)]
        writer.writerows(chunk)