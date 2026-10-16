from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


#Lightweight structure used to send dataset metadata to preview UI
#Keeps profile logic separate from routers so preview rendering stays clean
@dataclass
class DatasetProfile:
    columns: list[str]
    dtypes: dict[str, str]
    row_count: int
    col_count: int
    sample_rows: list[dict[str, Any]]


def build_dataset_profile_from_df(df: pd.DataFrame, limit: int = 10) -> DatasetProfile:
    #Creates a DatasetProfile from an existing pandas DataFrame
    #Used by preview routes after CSV loading is handled elsewhere

    #Normalize column names to strings so templates behave consistently
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    #Basic structural information used by preview page
    row_count = int(len(df))
    col_count = int(len(df.columns))

    #Convert pandas dtypes into readable strings for display
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}

    #Take a small subset of rows so preview loads quickly even for large datasets
    #Replace NaN values with None so JSON and Jinja templates render safely
    sample = df.head(limit)
    sample = sample.where(pd.notnull(sample), None)
    sample_rows = sample.to_dict(orient="records")

    return DatasetProfile(
        columns=list(df.columns),
        dtypes=dtypes,
        row_count=row_count,
        col_count=col_count,
        sample_rows=sample_rows,
    )


def build_dataset_profile(csv_path: Path, limit: int = 10, dataset_name: str | None = None) -> DatasetProfile:
    #Wrapper function that builds a profile directly from a CSV path
    #If dataset_name exists we reuse the shared loader so uploads and built-in datasets behave the same

    if dataset_name:
        #Local import avoids circular dependency with dataset_files module
        from .dataset_files import load_dataset_df
        df = load_dataset_df(dataset_name)
        return build_dataset_profile_from_df(df, limit=limit)

    #Fallback path for simple CSV files without special loading rules
    df = pd.read_csv(csv_path, low_memory=False)
    return build_dataset_profile_from_df(df, limit=limit)