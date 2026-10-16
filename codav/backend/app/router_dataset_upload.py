from __future__ import annotations
import os
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from .db import get_db
from .models import Dataset


router = APIRouter()

#Uploads it to the correct directory
BACKEND_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BACKEND_ROOT / "uploads"

#To make sure the folder actaully exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def build_safe_filename(original_name: str) -> str:
    #This is to generate a safe file name to prevent files from being overwritten
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #Removes unwanted characters
    cleaned = "".join(
        c for c in original_name if c.isalnum() or c in ("_", "-", ".")
    )

    return f"{timestamp}_{cleaned}"


#Post mapping for uploading the dataset
@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    domain: str | None = Form(None),
    description: str | None = Form(None),
    db: Session = Depends(get_db),
):

    #File validation ensuring only csv files and not clicking upload without anything attached
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    #Generates a safe path for the file
    safe_name = build_safe_filename(file.filename)
    save_path = UPLOAD_DIR / safe_name

    #Saves the file to the disk
    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    #This inserts the dataset row
    dataset = Dataset(
        name=name,
        domain=domain,
        source="User upload",
        description=description,
        file_path=f"uploads/{safe_name}",  
        original_filename=file.filename,
        is_user_uploaded=True,
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    #This is to redirect back to datasets page and then keeps the behaviour consistent with CODAV flow.
    return RedirectResponse(
        url="/datasets",
        status_code=303,
    )