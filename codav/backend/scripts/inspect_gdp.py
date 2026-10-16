from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "gdp.csv"
print("Looking for:", CSV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"The CSV is not found at {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="latin1")
print("\nCSV has loaded")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nHead:")
print(df.head(5).to_string(index=False))