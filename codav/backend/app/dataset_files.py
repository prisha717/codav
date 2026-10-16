#Central place where CODAV finds and loads CSV files into pandas
#This is used by both the preview page and the visualise page in main.py
#By keeping file loading here, I only need to fix loading rules in one place
from __future__ import annotations
from pathlib import Path
import pandas as pd

#Project root folder
#This works even when the file is imported from inside app/
ROOT = Path(__file__).resolve().parents[2]

#Folder for curated datasets that ship with CODAV
DATA_DIR = ROOT / "data"

#Folder for user uploads saved by the upload route in main.py
UPLOAD_DIR = ROOT / "backend" / "uploads"

#Maps a dataset name shown in the UI to a CSV filename in backend/data/
#If a dataset is not in this map, CODAV tries a simple "guess" filename using the dataset name
DATASET_FILE_MAP = {
    "World Bank GDP (Time Series)": "world_bank_gdp_timeseries.csv",
    "World Bank Inflation (Time Series)": "world_bank_inflation_timeseries.csv",
    "OECD Inflation": "global_inflation_data.csv",
    "Global Superstore": "Global_Superstore2.csv",
    "Store Transactions": "products.csv",
}

#These datasets are known to have extra metadata lines at the top of the CSV
#So we need to detect the real header row before reading
WORLD_BANK_TIME_SERIES = {
    "World Bank GDP (Time Series)",
    "World Bank Inflation (Time Series)",
}

def get_csv_path_for_dataset(dataset_name: str, file_path: str | None = None) -> Path:
    #Returns the CSV path for a dataset
    #For preview page and visualise page logic in main.py via load_dataset_df
    #If file_path is provided, treat it as a user uploaded file under backend/uploads/
    #Otherwise treat it as a curated dataset under backend/data/
    #User uploaded dataset path
    if file_path:
        #file_path is stored like "uploads/abc123_file.csv"
        #We only want the filename part so users cannot escape the uploads folder
        safe_name = Path(file_path).name
        candidate = UPLOAD_DIR / safe_name
        if not candidate.exists():
            raise FileNotFoundError(f"Uploaded CSV not found at {candidate}")
        return candidate

    #Curated dataset path using the name -> filename map
    filename = DATASET_FILE_MAP.get(dataset_name)

    if not filename:
        #Fallback for datasets that are not in the map
        #This keeps the project flexible if I add new built-in CSVs later
        guessed = dataset_name.strip().lower().replace(" ", "_") + ".csv"
        candidate = DATA_DIR / guessed
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No CSV mapping for dataset '{dataset_name}' "
            f"Add it to DATASET_FILE_MAP in backend/app/dataset_files.py"
        )

    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    return path

def _detect_world_bank_header_row(csv_path: Path) -> int:
    #World Bank time-series CSVs often include metadata rows before the real header
    #This function scans the first part of the file to find the header row
    expected = ("Country Name", "Country Code", "Indicator Name", "Indicator Code")

    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for i in range(0, 60):
            line = f.readline()
            if not line:
                break
            if all(k in line for k in expected):
                return i

    raise ValueError(f"Could not find World Bank header row in {csv_path}")

def _read_csv_with_fallback(csv_path: Path) -> pd.DataFrame:
    #Tries UTF-8 first, then latin1 as some public datasets download with different encodings
    #This avoids common crashes when reading Kaggle or World Bank CSVs
    try:
        return pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin1", low_memory=False)

def load_dataset_df(dataset_name: str, file_path: str | None = None) -> pd.DataFrame:
    #Loads a dataset into pandas. Used by: GET /datasets/{dataset_id}/visualise-page in main.py and any other backend logic that needs the dataset as a dataframe
    #Parameters: dataset_name is the display name stored in MySQL. file_path is only set for user uploads (stored in MySQL as Dataset.file_path)
    csv_path = get_csv_path_for_dataset(dataset_name=dataset_name, file_path=file_path)

    #World Bank time series needs a detected header row because metadata appears first
    if dataset_name in WORLD_BANK_TIME_SERIES:
        header_row = _detect_world_bank_header_row(csv_path)

        #Read the CSV starting from the detected header row
        #Use encoding fallback for reliability
        try:
            df = pd.read_csv(
                csv_path,
                skiprows=header_row,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                csv_path,
                skiprows=header_row,
                encoding="latin1",
                low_memory=False,
            )
        return df

    #Default read for most datasets
    return _read_csv_with_fallback(csv_path)