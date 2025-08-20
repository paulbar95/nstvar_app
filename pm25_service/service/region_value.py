import csv

def get_region_value(region: str, scenario: str):
    with open("data/pm25_region_values.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["region"] == region and row["scenario"] == scenario:
                return float(row["value"])
    return None
