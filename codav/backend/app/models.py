from __future__ import annotations
import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from .base import Base

#ORM model for datasets table. This represents every dataset shown in the CODAV UI.
# Both built in datasets and user uploads use the same table so the frontend does not need separate logic.
class Dataset(Base):
    __tablename__ = "datasets"
    # PRIMARY KEY: Unique id used in URLs like: /datasets/{dataset_id}/preview-page
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    #BASIC DATASET INFO:Dataset name shown in the UI
    #I keep this NOT NULL because every dataset must have a display name
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    #Optional metadata fields, I allow NULL because user uploads might not include full information.
    domain: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
        index=True,
    )

    source: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    #UPLOAD-RELATED METADATA: file_path stores where the CSV lives on disk
    #Built in datasets may leave this NULL because they are resolved via dataset_files.py
    file_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    #Original filename from the user upload, mainly for display and audit not for loading the file
    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    #Flag to distinguish curated datasets from uploads
    #The UI treats them the same, but this helps backend logic later
    is_user_uploaded: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="0",
        index=True,
    )
    #Automatically set by MySQL when a dataset is created
    #timezone=False because MySQL DATETIME does not store timezone info
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    #Links this dataset to preview rows stored in dataset_rows table
    #Used by preview routes when the CSV file is unavailable
    rows = relationship(
        "DatasetRow",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    #Helpful for debugging in the terminal/logs
    def __repr__(self) -> str:
        return (
            f"Dataset(id={self.id!r}, name={self.name!r}, "
            f"is_user_uploaded={self.is_user_uploaded!r})"
        )

#ORM model for storyboard containers used to group saved visualisations
class Storyboard(Base):
    __tablename__ = "storyboards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #User-defined storyboard title
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    #Timestamp automatically created by MySQL
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )

    #Relationship to slides belonging to this storyboard
    slides = relationship(
        "StoryboardSlide",
        back_populates="storyboard",
        cascade="all, delete-orphan",
        order_by="StoryboardSlide.order_index"
    )

# ORM model representing a single slide inside a storyboard
class StoryboardSlide(Base):
    __tablename__ = "storyboard_slides"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True
    )

    # Reference to storyboard container
    storyboard_id: Mapped[int] = mapped_column(
        ForeignKey("storyboards.id", ondelete="CASCADE"),
        nullable=False
    )

    # Reference to visualisation ID stored in visualisations table
    # We store only the ID instead of creating an ORM relationship
    visualisation_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Stored HTML snapshot of the chart (Plotly output)
    chart_html: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Stored Plotly figure JSON for reliable export
    chart_json: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Auto-generated caption from CODAV insights
    caption: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Optional user annotation
    user_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True
    )

    # Controls slide order in the storyboard
    order_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )

    # Relationship back to storyboard container
    storyboard = relationship(
        "Storyboard",
        back_populates="slides"
    )