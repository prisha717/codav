#Pydantic schemas used to define API response structures
#These act as a contract between backend and client so responses stay consistent
#Also helps Swagger automatically document endpoints

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict


#Represents dataset metadata returned by API endpoints
#Used when listing datasets or fetching a single dataset
class DatasetOut(BaseModel):

    #Allows Pydantic to read values directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str | None = None
    created_at: datetime


#Represents preview response containing a small subset of dataset rows
#Keeps payload lightweight so frontend loads quickly
class DatasetPreviewOut(BaseModel):
    dataset_id: int
    limit: int
    rows: list[dict[str, Any]]