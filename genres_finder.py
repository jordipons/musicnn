import csv

with open("fma\\data\\fma_metadata\\genres.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        if row["parent"] == "0":
            print(row["title"], end=", ")