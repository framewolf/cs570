import glob
import json
import pandas as pd

rows = []

for path in glob.glob("results/**/*.json", recursive=True):

    with open(path, "r") as f:
        rows.append(json.load(f))

df = pd.DataFrame(rows)

print(df)

df.to_csv("results_summary.csv", index=False)
