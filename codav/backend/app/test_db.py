from sqlalchemy import text
from .db import engine
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1")).scalar()
    print("DB OK:", result)