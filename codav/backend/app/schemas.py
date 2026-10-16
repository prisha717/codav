from pydantic import BaseModel
from datetime import datetime
from typing import Dict

class DatasetOut(BaseModel):
    id: int
    name: str
    domain: str
    source: str | None = None
    description: str | None = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class InsightOut(BaseModel):
    visualisation_id: int
    summary: Dict[str, int]
    missing_values: Dict[str, int]