from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import datetime


class VisualisationCreate(BaseModel):
    dataset_id: int
    chart_type: str = Field(min_length=1, max_length=50)
    title: Optional[str] = Field(default=None, max_length=200)
    spec: Dict[str, Any]


class VisualisationOut(BaseModel):
    id: int
    dataset_id: int
    chart_type: str
    title: Optional[str]
    spec: Dict[str, Any]
    created_at: datetime.datetime

    class Config:
        from_attributes = True