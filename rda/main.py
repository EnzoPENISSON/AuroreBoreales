def generate_files(base_path, start_year, end_year, start_month=1, end_month=12):
    files = []

    for year in range(start_year, end_year + 1):
        for month in range(start_month if start_year == year else 1, end_month + 1 if end_year == year else 13):
            files.append(f"{base_path}/{year}/{year}{month:02}_cleaned.csv")

    return files

kp_files = ["data/kp-compiled/kp_cleaned.csv"]
x_files = generate_files("data/mag-kiruna-compiled", 2007, 2024, 1, 6)

# Ace
solar_wind_files = generate_files(
    "data/solarwinds-ace-compiled",
    2007,
    2016,
    1,
    7
)

# Discover
solar_wind_files += generate_files(
    "data/solarwinds-dscovr-compiled",
    2016,
    2024,
    7,
    6
)

def run():
    print(x_files)
    print(solar_wind_files)


def smooth(inputs):
    avg = 0

    for x in inputs:
        avg += x

    avg /= len(inputs)

    return avg


def merge():
    pass


if __name__ == '__main__':
    run()