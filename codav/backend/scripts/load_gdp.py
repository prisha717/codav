import sys
import json
from pathlib import Path
from sqlalchemy import text
BACKEND_ROOT = Path(__file__).resolve().parents[1]  # points to /backend
sys.path.append(str(BACKEND_ROOT))
import pandas as pd 
from app.db import get_db #I am resuing the same database logic used by the FastAPI app to keep it consistent with the backend environment
from app.models import Dataset #The SQLAlchemy ORM model to represent the datasets table
import app.register_models

#Configuring the metadata, keeping it all in one place so the script is easily adaptable for new datasets
DATASET_NAME = "OECD GDP (CSV)"
DATASET_DOMAIN = "finance" #Ensure it matches the ones in MySQL uch as retail, finance etc
DATASET_SOURCE = "OECD"
DATASET_DESCRIPTION = "The GDP dataset is loaded from a local CSV file for CODAV testing"
PREVIEW_LIMIT = 20 #This is the number of rows to store for the /datasets/{id}/preview

#This is to resolve the project root so the __file__ is this file
ROOT = Path(__file__).resolve().parents[2] #parents[2] goes to the repository root folder ie CODAV

#Te dataset is stored locally and is ignored by git to avoid issues
CSV_PATH = ROOT / "data" / "gdp.csv"

def main() -> None:
    print("Currently looking for:", CSV_PATH) #1.To confirm file exists to identify failures ear;y
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV cannot be found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, encoding="latin1") #2.Read csv file to confirm it is valid, used latin1 to help with Windows
    print("CSV. Rows:", len(df), "Cols:", len(df.columns))
    db_gen = get_db() #3.Open up the database session, used by fastapi and i reuse it to make/close a session the way api does
    db = next(db_gen)

    try: #4.Prevents duplicate by running this to prevent repeated rows with the same dataset name
        existing = db.query(Dataset).filter(Dataset.name == DATASET_NAME).first()

        if existing:
            dataset = existing
            print(f"The dataset already exists (id={dataset.id}). Metadata insert skipped.")
        else:
            dataset = Dataset(
                name=DATASET_NAME,
                domain=DATASET_DOMAIN,
                source=DATASET_SOURCE,
                description=DATASET_DESCRIPTION,
            )
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            print(f"Inserted the dataset id={dataset.id}, name='{dataset.name}'")
        #Check if preview rows already exist for this dataset
        existing_preview_count = db.execute(
            text("SELECT COUNT(*) FROM dataset_rows WHERE dataset_id = :dataset_id"),
            {"dataset_id": dataset.id},
        ).scalar()

        if existing_preview_count and existing_preview_count > 0:
            print(f"Preview rows already exist for dataset_id={dataset.id}. Skipping preview insert.")
        else:
            #Take only the first PREVIEW_LIMIT rows to keep storage small and predictable
            preview_df = df.head(PREVIEW_LIMIT)
            # Convert NaN values to None so JSON is valid for MySQL
            preview_df = preview_df.where(pd.notna(preview_df), None)
            raw_records = preview_df.to_dict(orient="records")
            records = [
                {k: (None if pd.isna(v) else v) for k, v in row.items()}
                for row in raw_records
            ]


            insert_sql = text("""
                INSERT INTO dataset_rows (dataset_id, row_index, data)
                VALUES (:dataset_id, :row_index, :data)
            """)

            for i, row in enumerate(records):
                db.execute(
                    insert_sql,
                    {
                        "dataset_id": dataset.id,
                        "row_index": i,
                        "data": json.dumps(row),
                    },
                )
            db.commit()
            print(f"Inserted {len(records)} preview rows into dataset_rows for dataset_id={dataset.id}")

    except Exception as e:
        db.rollback()
        print(f"The ingestion has failed thus transaction rolled back")
        raise e #If anything failes after transation has started then rollback to prevent partial or incosistent writes
    finally:
        #Close session generator cleanly, mirroring fastapi's dependency cleanup
        try:
            next(db_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    main()