import pandas as pd
import re
from typing import Dict, Any


#Handles preparation of data structures used by visualisation logic
#Acts as a lightweight adapter between raw datasets and chart-ready formats
class Visualisation:

    #Builds a generic time-series structure by detecting year-like columns
    #Designed mainly for World Bank style datasets where years appear in headers
    @staticmethod
    def time_series(df: pd.DataFrame) -> Dict[str, Any]:

        year_cols = []
        year_labels = []

        #Scan column names and extract any 4 digit year pattern
        #Supports formats like 1960 1960 space or 1960 [YR1960]
        for col in df.columns:
            col_str = str(col)

            match = re.search(r"\b(19|20)\d{2}\b", col_str)
            if match:
                year_cols.append(col)
                year_labels.append(match.group(0))

        #Fail early if no usable time axis exists
        if not year_cols:
            raise ValueError(
                "No suitable year columns found for time-series visualisation"
            )

        #Convert values to numeric and average across rows to produce a single series
        #Mean aggregation keeps beginner visualisations simple and readable
        series = (
            df[year_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .dropna()
        )

        #Return lightweight chart payload instead of a full Plotly object
        return {
            "type": "time_series",
            "x": year_labels,
            "y": list(series.values),
        }

    #Simple fallback preview when dataset cannot be treated as a time series
    #Used to keep the UI functional even when no chart is generated
    @staticmethod
    def table_preview(df: pd.DataFrame, limit: int = 10) -> Dict[str, Any]:
        return {
            "type": "table",
            "columns": list(df.columns),
            "rows": df.head(limit).fillna("").to_dict(orient="records"),
        }

    #Entry point used by routes to decide which visualisation structure to build
    #Uses dataset naming convention to determine if time-series logic should run
    @staticmethod
    def build(dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        if "Time Series" in dataset_name:
            return Visualisation.time_series(df)

        #Fallback behaviour keeps system robust for heterogeneous datasets
        return Visualisation.table_preview(df)