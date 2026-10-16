from sqlalchemy.orm import Session
import pandas as pd

#Import SQLAlchemy model for stored visualisations
#Used to link a chart back to its dataset_id so insights can be generated
from .model_visualisation import Visualisation

#Reuse existing preview helper instead of loading full CSV again
#Keeps insight generation lightweight and consistent with preview logic
from .crud_datasets import get_dataset_preview_rows


def generate_insights(db: Session, visualisation_id: int, limit: int = 500) -> dict | None:
    #Generates simple statistical insights for a visualisation
    #Flow:
    #1 Fetch visualisation from database
    #2 Use dataset_id to retrieve preview rows
    #3 Convert rows to pandas DataFrame and compute basic metrics

    #Fetch visualisation directly from DB so logic stays contained inside CRUD layer
    visualisation = (
        db.query(Visualisation)
        .filter(Visualisation.id == visualisation_id)
        .first()
    )

    #Return None so API layer can decide how to handle missing visualisation
    if visualisation is None:
        return None

    #Retrieve a limited number of preview rows instead of full dataset for performance reasons
    rows = get_dataset_preview_rows(db, visualisation.dataset_id, limit)

    #If dataset has no rows return empty insight structure rather than failing
    if not rows:
        return {
            "visualisation_id": visualisation_id,
            "summary": {"rows": 0, "columns": 0},
            "missing_values": {},
        }

    #Convert JSON preview rows into a pandas DataFrame for quick analysis
    df = pd.DataFrame(rows)

    #Basic structural summary used by UI to give context about the dataset
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    #Count missing values per column to highlight potential data quality issues
    missing_values = {
        str(col): int(count)
        for col, count in df.isna().sum().to_dict().items()
    }

    #Return lightweight insight payload that can be expanded later without changing schema
    return {
        "visualisation_id": visualisation_id,
        "summary": summary,
        "missing_values": missing_values,
    }