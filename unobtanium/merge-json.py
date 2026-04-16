import pandas as pd
import json

def merge(file1, file2, final):
    # Charger fichiers
    with open("data/json/" + file1) as f:
        data1 = json.load(f)

    with open("data/json/" + file2) as f:
        data2 = json.load(f)

    columns_to_keep = ["time_tag", "speed", "density", "bt", "bz_gsm"]

    # Convertir en DataFrame
    df1 = pd.DataFrame(data1[1:], columns=data1[0])
    df2 = pd.DataFrame(data2[1:], columns=data2[0])

    # Convertir les dates (important pour merge fiable)
    df1["time_tag"] = pd.to_datetime(df1["time_tag"])
    df2["time_tag"] = pd.to_datetime(df2["time_tag"])

    # Merge sur time_tag
    merged = pd.merge(df1, df2, on="time_tag", how="inner")
    filtered = merged[columns_to_keep]

    # Sauvegarde
    filtered.to_json("data/json/" + final, orient="records", date_format="iso")


if __name__ == "__main__":
    merge("mag-1h.json", "plasma-storm-1h.json", "merged.json")
    merge("mag-storm-1h.json", "plasma-storm-1h.json", "merged-storm.json")