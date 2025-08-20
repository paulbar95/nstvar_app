import csv

def get_threshold(scenario: str):
    with open("data/pm25_thresholds.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["scenario"] == scenario:
                return float(row["threshold"])
    return None
