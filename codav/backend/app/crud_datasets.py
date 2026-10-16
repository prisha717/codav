#Dataset CRUD operations
#This module handles database queries related to datasets
#Keeping CRUD logic separate from routes helps keep the API layer clean
#It also makes the backend easier to maintain and test during development

from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models import Dataset
from app.model_dataset_row import DatasetRow

def get_dataset_by_id(db: Session, dataset_id: int) -> Dataset | None:
    #Retrieves a single dataset using its primary key
    #Used across preview, visualisation and API routes so logic stays reusable
    #Returns None instead of raising an error so the router can decide how to handle missing datasets

    stmt = select(Dataset).where(Dataset.id == dataset_id)
    return db.execute(stmt).scalar_one_or_none()

def get_dataset_preview_rows(
    db: Session,
    dataset_id: int,
    limit: int,
) -> list[dict]:
    #Returns a small subset of rows stored in DatasetRow for preview purposes
    #Rows are ordered by row_index so the original CSV order is preserved
    #This acts as a fallback preview when a CSV file cannot be loaded directly

    stmt = (
        select(DatasetRow.data)
        .where(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index.asc())
        .limit(limit)
    )

    #Each row is stored as JSON so we extract the dictionary from the SQL result
    return [row[0] for row in db.execute(stmt).all()]