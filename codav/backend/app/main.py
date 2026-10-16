import json
import re
import uuid
from io import BytesIO
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Query, File, UploadFile, Form, Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, FileResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import tempfile
from sqlalchemy.orm import Session, selectinload
from .db import get_db, engine
from .base import Base
from . import register_models
from .models import Dataset
from .models import Storyboard
from .models import StoryboardSlide
from .schemas import DatasetOut
from .schemas_dataset import DatasetPreviewOut
from .crud_datasets import get_dataset_by_id, get_dataset_preview_rows
from .schemas_insight import InsightOut
from .crud_insights import generate_insights
from .schemas_visualisation import VisualisationCreate, VisualisationOut
from .dataset_files import get_csv_path_for_dataset, load_dataset_df
from .dataset_profile import build_dataset_profile
from .router_dataset_upload import router as upload_router #Did not add @app.post("/datasets/upload") else it would crash
from .charting import ChartSpec, infer_time_series_spec, infer_generic_fallback_spec, make_chart_html, make_chart_json, find_year_columns
from sqlalchemy import text, func
from .dataset_linking import shared_columns_exact, suggest_links, suggestions_to_dict
from .insights import generate_insights
from app.models import Storyboard, StoryboardSlide
from app.explanation import build_chart_explanation_dict
import plotly.io as pio
import plotly.graph_objects as go

#This sets up the uploads folder inside backend/uploads, using Path keeps the file handling clean. 
REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = REPO_ROOT / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024 #Has a maximum file size of 10MB, avoids large files from crashing the system.
ALLOWED_EXTENSIONS = {".csv"}

def _safe_stem(filename: str) -> str: #Convert file name to safer version for storage and display.
    stem = Path(filename).stem.strip().lower() #Removes spces/symbols so no path or formatting issues.
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^a-z0-9_\-]+", "", stem)
    return stem or "uploaded_dataset"

def _make_unique_dataset_name(db: Session, base_name: str) -> str: #Uploaded datasets don't clash with existing dataset name in MySQL
    candidate = base_name
    i = 2
    while db.query(Dataset).filter(Dataset.name == candidate).first() is not None:
        candidate = f"{base_name}_{i}"
        i += 1 #Example, Sales Data to Sales Data_2 if it already exists
    return candidate

def _read_upload_bytes(upload: UploadFile) -> bytes: #Read uploaded file into memory and stops large files
    data = upload.file.read() #Does it now to keep the rest of the system running smoothly.
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10MB).")
    return data

def _basic_csv_validation(data: bytes) -> None: #Basic validation to stop empty or CSV files.
    try:
        df = pd.read_csv(BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV file (could not parse).")

    if df.shape[1] < 1:
        raise HTTPException(status_code=400, detail="CSV must have at least one column.")
    if df.shape[0] < 1:
        raise HTTPException(status_code=400, detail="CSV must have at least one row.")

def reshape_world_bank_timeseries(df: pd.DataFrame, country_name: str | None = None) -> pd.DataFrame:
    if country_name and "Country Name" in df.columns: #Converts time-series format to simple long format
        df = df[df["Country Name"] == country_name] #Output becomes 2 key columns: year and value for cleaner x/y structure
    else: #If no chosen country, take first usable row with actual year values.
        df = df.dropna(how="all", subset=df.columns[4:]).head(1)
    if df.empty:
        raise ValueError("No country data available to reshape")
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("No year columns found in World Bank dataset")
    long_df = df.melt(value_vars=year_cols, var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"]).sort_values("year")
    long_df["year"] = long_df["year"].astype(int)
    return long_df

def reshape_world_bank_gdp_timeseries(df: pd.DataFrame, country_name: str = "World") -> pd.DataFrame: #Same as before
    if "Country Name" in df.columns:
        pick = df[df["Country Name"] == country_name]
        if pick.empty:
            pick = df.dropna(how="all", subset=df.columns[4:]).head(1)
        df = pick
    if df.empty:
        raise ValueError("No country data available to reshape")
    year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    if not year_cols:
        raise ValueError("No year columns found in World Bank GDP time-series dataset")
    long_df = df.melt(value_vars=year_cols, var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"]).sort_values("year")
    long_df["year"] = long_df["year"].astype(int)
    return long_df

def reshape_oecd_inflation_to_single_series(df: pd.DataFrame) -> pd.DataFrame: #OECD inflation to 1 time-series if multiple indicators for 1 thing, for beginner it produces 1 clean chart
    if df is None or df.empty: 
        raise ValueError("OECD dataset is empty")
    year_cols = find_year_columns(df)
    if not year_cols:
        raise ValueError("No year columns found (expected headers like 1960, 1961, ...)")
    meta_cols: list[str] = []
    for c in ["country_name", "indicator_name"]:
        if c in df.columns:
            meta_cols.append(c)
    long_df = df.melt(
        id_vars=meta_cols if meta_cols else None,
        value_vars=year_cols,
        var_name="year",
        value_name="value",
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"])
    if long_df.empty:
        raise ValueError("No usable numeric values after reshaping")
    if "indicator_name" in long_df.columns: #If multiple indicators, pick most common to keep default view consistent.
        top_indicator = long_df["indicator_name"].value_counts().idxmax()
        long_df = long_df[long_df["indicator_name"] == top_indicator]
    series_df = long_df.groupby("year", as_index=False)["value"].mean().sort_values("year")
    series_df["year"] = series_df["year"].astype(int)
    return series_df

def build_count_time_series(df: pd.DataFrame, date_col: str) -> pd.DataFrame: #Transaction-style datasets with dates but no clear numeric measure.
    tmp = df.copy() 
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", utc=False)
    tmp = tmp.dropna(subset=[date_col]) #Remove rows when date is not understood
    if tmp.empty:
        raise ValueError("No valid dates found for counting time-series") #When no usable dates exist
    tmp["date_bucket"] = tmp[date_col].dt.date #Reduces timestamps to calendar date and groups events on same day together'
    out = tmp.groupby("date_bucket", as_index=False).size() #Counts how many rows exist per day to give transaction per day instead of not plotting. at all.
    out = out.rename(columns={"date_bucket": "date", "size": "value"}) #Renames the columns to match ChartSpec format used in CODAV.
    out["date"] = pd.to_datetime(out["date"]) #Convert back to datetime so Plotly treats it as a time axis
    return out.sort_values("date") #Sorts chronologically so charts appear clean

def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None: #Datasets name columns differently so this checks likely names
    lower_map = {str(c).strip().lower(): c for c in df.columns} #Compare names
    for want in candidates: #Loops through candidates in order and returns the first match
        if want.strip().lower() in lower_map:
            return lower_map[want.strip().lower()]
    return None #If none match then return None

def prepare_expert_chart_df( #Prepares dataframe for expert mode using x and y columns selected by user.
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    chart_type: str,
) -> pd.DataFrame:
    tmp = df.copy()
    if x_col not in tmp.columns or y_col not in tmp.columns:
        raise ValueError("Selected columns not found in dataset")
    # For bar charts, only aggregate if y is numeric and x is categorical
    if chart_type == "bar":
        is_x_categorical = (
            pd.api.types.is_object_dtype(tmp[x_col])
            or pd.api.types.is_string_dtype(tmp[x_col])
            or pd.api.types.is_categorical_dtype(tmp[x_col])
        )
        y_numeric = pd.to_numeric(tmp[y_col], errors="coerce")
        #Only do mean aggregation when y actually has numeric values
        if is_x_categorical and y_numeric.notna().any():
            tmp = tmp[[x_col, y_col]].copy()
            tmp[y_col] = y_numeric
            tmp = tmp.dropna(subset=[x_col, y_col])
            if not tmp.empty:
                tmp = (
                    tmp.groupby(x_col, as_index=False)[y_col]
                    .mean()
                    .sort_values(by=y_col, ascending=False)
                )
                return tmp
    tmp = tmp.dropna(subset=[x_col, y_col]) #Fallabck - keep non-null rows for chosen columns
    return tmp

Base.metadata.create_all(bind=engine) #Creates database tables when app starts
app = FastAPI() #Main FASTAPI instance
templates = Jinja2Templates(directory="app/templates") #Jinja template to render HTML page
app.add_middleware(SessionMiddleware, secret_key="5e2ffb42729e6d71b9103da3da55820c6aacd6606f63fa765984cf6586185ddf")

@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)

    dataset_count = db.query(Dataset).count()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "dataset_count": dataset_count
        }
    )

@app.get("/datasets-page") #Dataset List: displays the datasets in MySQL, built-in and uploaded
def datasets_page(request: Request, db: Session = Depends(get_db)):
    datasets = db.query(Dataset).order_by(Dataset.id).all()
    return templates.TemplateResponse("datasets.html", {"request": request, "datasets": datasets})

@app.get("/datasets/{dataset_id}/preview-page") #Dataset Preview: User inspects rows and columns before chossing visualisation
def dataset_preview_page( #Supports explainability aim as users can see raw data first
    dataset_id: int,
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db, dataset_id) #Looks up dataset in MySQL by its id
    if dataset is None: #If they have wrong URL/dataset deleted then error shown
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.file_path: #If dataset has a stored file path, try reading actual CSV first
        try:
            df = load_dataset_df(dataset.name, dataset.file_path)
            sample = df.head(limit)
            rows = sample.where(pd.notna(sample), None).to_dict(orient="records")
            columns = list(sample.columns)
            dtypes = {c: str(df[c].dtype) for c in df.columns}
            return templates.TemplateResponse(
                "dataset_preview.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "rows": rows,
                    "columns": columns,
                    "dtypes": dtypes,
                    "row_count": int(df.shape[0]),
                    "col_count": int(df.shape[1]),
                    "limit": limit,
                    "csv_path": dataset.file_path,
                },
            )
        except Exception as e: #Returns an error on the page instead of crashing the system
            return templates.TemplateResponse(
                "dataset_preview.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "rows": [],
                    "columns": [],
                    "dtypes": {},
                    "row_count": None,
                    "col_count": None,
                    "limit": limit,
                    "csv_path": dataset.file_path,
                    "error": f"Preview failed: {e}",
                },
            )
    try: #Built-in datasets, csv file is loaded to give better preview to calculate types, rows etc
        csv_path = get_csv_path_for_dataset(dataset.name, dataset.file_path)
        profile = build_dataset_profile(csv_path, limit=limit, dataset_name=dataset.name)
        return templates.TemplateResponse( #Preview page with metadata, sample rows and column info
            "dataset_preview.html",
            {
                "request": request,
                "dataset": dataset,
                "rows": profile.sample_rows,
                "columns": profile.columns,
                "dtypes": profile.dtypes,
                "row_count": profile.row_count,
                "col_count": profile.col_count,
                "limit": limit,
                "csv_path": str(csv_path),
            },
        )
    except FileNotFoundError: #If csv not found, use preview rows in db instead so page works even if missing file
        rows = get_dataset_preview_rows(db, dataset_id, limit)
        columns = sorted({k for row in rows for k in row.keys()}) if rows else [] #Build a list of columns based on the keys found in the preview rows
        return templates.TemplateResponse(
            "dataset_preview.html",
            {
                "request": request,
                "dataset": dataset,
                "rows": rows,
                "columns": columns,
                "dtypes": {},
                "row_count": None,
                "col_count": None,
                "limit": limit,
                "csv_path": None,
            },
        )
    except Exception as e: 
        return templates.TemplateResponse(
            "dataset_preview.html",
            {
                "request": request,
                "dataset": dataset,
                "rows": [],
                "columns": [],
                "dtypes": {},
                "row_count": None,
                "col_count": None,
                "limit": limit,
                "csv_path": None,
                "error": f"Preview failed: {e}",
            },
        )

@app.get("/datasets/preview-all-page") #Preview All Datasets: Quick overview of multiple datsets at once to browse 
def preview_all_datasets_page(
    request: Request,
    limit: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
): #Gets datasets from MySQL
    datasets = db.query(Dataset).order_by(Dataset.id.asc()).all()
    preview_cards = [] #Dataset becomes a card on the page
    for dataset in datasets:
        try: #Load csv
            csv_path = get_csv_path_for_dataset(dataset.name, dataset.file_path)
            profile = build_dataset_profile(csv_path, limit=limit, dataset_name=dataset.name)
            preview_cards.append(
                {
                    "dataset": dataset,
                    "csv_path": str(csv_path),
                    "row_count": profile.row_count,
                    "col_count": profile.col_count,
                    "columns": profile.columns,
                    "dtypes": profile.dtypes,
                    "rows": profile.sample_rows,
                }
            )
        except Exception as e: #If one dataset fails, page still loads
            preview_cards.append({"dataset": dataset, "error": str(e)})
    return templates.TemplateResponse(
        "datasets_preview_all.html",
        {"request": request, "preview_cards": preview_cards, "limit": limit},
    )

@app.get("/datasets/{dataset_id}/visualise-page") #visualisation Page: Loads dataset, decides what chart to show and supports both mode
def dataset_visualise_page(
    dataset_id: int,
    request: Request,
    mode: str = Query("beginner", pattern="^(beginner|expert)$"),
    chart_type: str = Query("line", pattern="^(line|bar|scatter)$"),
    x_col: str | None = Query(None),
    y_col: str | None = Query(None),
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db, dataset_id) #Loads dataset metadata from MySQL using the id in the URL
    if not dataset:#Dataset identified using URL id then loads from mysql
        raise HTTPException(status_code=404, detail="Dataset not found")
    #This keeps the linking logic server-side and avoids hardcoding dataset options in the template
    all_datasets = db.query(Dataset).order_by(Dataset.id).all() #PList of datasets for linking UI so users select a second dataset to compare against
    storyboards = ( #Load existing storyboards so the user can save to one of them from the visualise page
        db.query(Storyboard)
        .order_by(Storyboard.created_at.desc(), Storyboard.id.desc())
        .all()
    )
    chart_explanation = None
    try:#Loads dataset into pandas to clean it before plotting using Plotly
        df = load_dataset_df(dataset.name, dataset.file_path)
    except Exception as e:#Loading fails error message
        return templates.TemplateResponse(
            "dataset_visualise.html",
            {
                "request": request,
                "dataset": dataset,
                "all_datasets": all_datasets,
                "storyboards": storyboards,
                "error": f"Failed to load dataset: {e}",
                "mode": mode,
                "chart_type": chart_type,
                "columns": [],
                "selected_x_col": x_col,
                "selected_y_col": y_col,
            },
        )
    if dataset.name == "World Bank GDP": #World Bank GDP: This was better shown as bar rather than time-series
        try:
            df_gdp = df.copy().dropna(how="all")#Removes empty rows
            economy_col = "Unnamed: 3" #Column names specific to dataset format
            value_col = "Gross domestic product 2017"
            if economy_col not in df_gdp.columns or value_col not in df_gdp.columns:
                raise ValueError("Expected GDP columns not found")
            df_gdp = df_gdp[[economy_col, value_col]].dropna() #Keeps 2 important columns and removes missing values
            df_gdp[value_col] = df_gdp[value_col].astype(str).str.replace(",", "", regex=False)#Cleans numeric field and converts to numbers to sort
            df_gdp[value_col] = pd.to_numeric(df_gdp[value_col], errors="coerce")
            df_gdp = df_gdp.dropna(subset=[value_col]).sort_values(by=value_col, ascending=False).head(10)
            import plotly.express as px #Plotly bar chart
            fig = px.bar(
                df_gdp,
                x=economy_col,
                y=value_col,
                title="World Bank GDP — Selected economies (2017)",
                labels={economy_col: "Economy", value_col: "GDP (USD, millions)"},
            )
            chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
            chart_json = fig.to_json()
            columns = list(df_gdp.columns)
            spec = ChartSpec( #Pass spec to UI to show what was plotted
                chart_type="bar",
                x_col=economy_col,
                y_col=value_col,
                title="World Bank GDP — Selected economies (2017)",
            )
            chart_explanation = build_chart_explanation_dict(
                plot_df=df_gdp,
                chart_type="bar",
                x_col=economy_col,
                y_col=value_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "chart_explanation": chart_explanation,
                    "error": None,
                    "mode": "beginner",
                    "chart_type": "bar",
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "error": f"GDP visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
            )

    if dataset.name == "World Bank Inflation (Time Series)": #World Bank Inflation (Time Series)
        try:
            columns = list(df.columns) 
            if mode == "beginner":
                chart_df = reshape_world_bank_timeseries(df, country_name="World")
                chosen_chart = "line"
                title_map = {
                    "line": "World Bank Inflation — World (Line)",
                    "bar": "World Bank Inflation — World (Bar)",
                    "scatter": "World Bank Inflation — World (Scatter)",
                }
                spec = ChartSpec(
                    chart_type=chosen_chart,
                    x_col="year",
                    y_col="value",
                    title=title_map[chosen_chart],
                )
            else: #Expert mode keeps original dataframe so user can choose their own columns
                chosen_chart = chart_type
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"World Bank Inflation ({chosen_chart.title()})",
                    )
                else: #Fallback before the user selects manual axes.
                    chart_df = reshape_world_bank_timeseries(df, country_name="World")
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col="year",
                        y_col="value",
                        title=f"World Bank Inflation ({chosen_chart.title()})",
                    )
            chart_html = make_chart_html(chart_df, spec)
            chart_json = make_chart_json(chart_df, spec)
            chart_explanation = build_chart_explanation_dict(
                plot_df=chart_df,
                chart_type=spec.chart_type,
                x_col=spec.x_col,
                y_col=spec.y_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "error": None,
                    "mode": mode,
                    "chart_type": chosen_chart,
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": None,
                    "error": f"Inflation visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
    )

    if dataset.name == "OECD Inflation": #OECD Inflation: Beginner mode has a simplified average time-series
        try:
            columns = list(df.columns)
            if mode == "beginner":
                chart_df = reshape_oecd_inflation_to_single_series(df)
                chosen_chart = "line"
                title_map = {
                    "line": "OECD Inflation — Mean across countries (Line)",
                    "bar": "OECD Inflation — Mean across countries (Bar)",
                    "scatter": "OECD Inflation — Mean across countries (Scatter)",
                }
                spec = ChartSpec(
                    chart_type=chosen_chart,
                    x_col="year",
                    y_col="value",
                    title=title_map[chosen_chart],
                )
            else: #Expert mode should use the original dataframe, not the averaged time series.
                chosen_chart = chart_type
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"OECD Inflation ({chosen_chart.title()})",
                    )
                else: #Fallback before the user selects manual axes.
                    chart_df = reshape_oecd_inflation_to_single_series(df)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col="year",
                        y_col="value",
                        title=f"OECD Inflation ({chosen_chart.title()})",
                    )
            chart_html = make_chart_html(chart_df, spec)
            chart_json = make_chart_json(chart_df, spec)
            chart_explanation = build_chart_explanation_dict(
                plot_df=chart_df,
                chart_type=spec.chart_type,
                x_col=spec.x_col,
                y_col=spec.y_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "error": None,
                    "mode": mode,
                    "chart_type": chosen_chart,
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": None,
                    "error": f"OECD visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
    )

    if dataset.name == "World Bank GDP (Time Series)":
        try:
            columns = list(df.columns)
            if mode == "beginner":
                chart_df = reshape_world_bank_gdp_timeseries(df, country_name="World")
                chosen_chart = "line"
                title_map = {
                    "line": "World Bank GDP (Time Series) — World (Line)",
                    "bar": "World Bank GDP (Time Series) — World (Bar)",
                    "scatter": "World Bank GDP (Time Series) — World (Scatter)",
                }
                spec = ChartSpec(
                    chart_type=chosen_chart,
                    x_col="year",
                    y_col="value",
                    title=title_map[chosen_chart],
                )
            else: #Expert mode should use the original dataframe, not only the reshaped [year, value] view.
                chosen_chart = chart_type
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"World Bank GDP ({chosen_chart.title()})",
                    )
                else: #Fallback before the user selects manual axes.
                    chart_df = reshape_world_bank_gdp_timeseries(df, country_name="World")
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col="year",
                        y_col="value",
                        title=f"World Bank GDP ({chosen_chart.title()})",
                    )
            chart_html = make_chart_html(chart_df, spec)
            chart_json = make_chart_json(chart_df, spec)
            chart_explanation = build_chart_explanation_dict(
                plot_df=chart_df,
                chart_type=spec.chart_type,
                x_col=spec.x_col,
                y_col=spec.y_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "error": None,
                    "mode": mode,
                    "chart_type": chosen_chart,
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": None,
                    "error": f"World Bank GDP time-series visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
    )

    if dataset.name == "Store Transactions":
        try:
            columns = list(df.columns)
            if mode == "beginner": #Try common timestamp names as datasets can vary.
                date_col = find_first_existing_column(df, ["timestamp", "date", "order date", "time"])
                if not date_col:
                    raise ValueError("No timestamp/date column found for Store Transactions")
                chart_df = build_count_time_series(df, date_col) #Build a daily transaction count series for the beginner-friendly default view.
                chosen_chart = "bar"
                title_map = {
                    "line": "Store Transactions — Transactions per day (Line)",
                    "bar": "Store Transactions — Transactions per day (Bar)",
                    "scatter": "Store Transactions — Transactions per day (Scatter)",
                }
                spec = ChartSpec(
                    chart_type=chosen_chart,
                    x_col="date",
                    y_col="value",
                    title=title_map[chosen_chart],
                )
            else: #Expert mode should use the original dataframe, not the aggregated time series.
                chosen_chart = chart_type
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"Store Transactions ({chosen_chart.title()})",
                    )
                else: #Fallback for expert mode before the user selects custom axes.
                    date_col = find_first_existing_column(df, ["timestamp", "date", "order date", "time"])
                    if not date_col:
                        raise ValueError("No timestamp/date column found for Store Transactions")
                    chart_df = build_count_time_series(df, date_col)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col="date",
                        y_col="value",
                        title=f"Store Transactions ({chosen_chart.title()})",
                    )
            chart_html = make_chart_html(chart_df, spec)
            chart_json = make_chart_json(chart_df, spec)
            chart_explanation = build_chart_explanation_dict(
                plot_df=chart_df,
                chart_type=spec.chart_type,
                x_col=spec.x_col,
                y_col=spec.y_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "error": None,
                    "mode": mode,
                    "chart_type": chosen_chart,
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": None,
                    "error": f"Store Transactions visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
        )

    if dataset.name == "Global Superstore":
        try:
            columns = list(df.columns) #Beginner mode uses a cleaned monthly summary so the first chart is easy to understand.
            if mode == "beginner":
                date_col = find_first_existing_column(df, ["order date", "order_date", "date"])
                if not date_col:
                    raise ValueError("No Order Date column found")
                sales_col = find_first_existing_column(df, ["sales"])
                profit_col = find_first_existing_column(df, ["profit"])
                metric_col = sales_col or profit_col
                tmp = df.copy() #Use a copy so the original dataframe is not modified.
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", utc=False)
                tmp = tmp.dropna(subset=[date_col])
                if metric_col: #If Sales or Profit exists, show monthly totals.
                    tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
                    tmp = tmp.dropna(subset=[metric_col])
                    if tmp.empty:
                        raise ValueError("No usable rows after cleaning date + metric")
                    tmp["date_bucket"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                    chart_df = (
                        tmp.groupby("date_bucket", as_index=False)[metric_col]
                        .sum()
                        .rename(columns={"date_bucket": "date", metric_col: "value"})
                        .sort_values("date")
                    )
                    metric_name = "Sales" if metric_col == sales_col else "Profit"
                    title_base = f"Global Superstore — {metric_name} per month"
                else: #Fallback: count orders per month if no numeric metric is available.
                    tmp["date_bucket"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                    chart_df = (
                        tmp.groupby("date_bucket", as_index=False)
                        .size()
                        .rename(columns={"date_bucket": "date", "size": "value"})
                        .sort_values("date")
                    )
                    title_base = "Global Superstore — Orders per month"
                chosen_chart = "bar"
                title_map = {
                    "line": f"{title_base} (Line)",
                    "bar": f"{title_base} (Bar)",
                    "scatter": f"{title_base} (Scatter)",
                }
                spec = ChartSpec(
                    chart_type=chosen_chart,
                    x_col="date",
                    y_col="value",
                    title=title_map[chosen_chart],
                )
            else: #Expert mode should use the original dataframe, not the pre-aggregated monthly series.
                chosen_chart = chart_type
                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
                    spec = ChartSpec(
                        chart_type=chosen_chart,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"Global Superstore ({chosen_chart.title()})",
                    )
                else: #If no manual selection has been made yet, fall back to a sensible default.
                    date_col = find_first_existing_column(df, ["order date", "order_date", "date"])
                    sales_col = find_first_existing_column(df, ["sales"])
                    profit_col = find_first_existing_column(df, ["profit"])
                    metric_col = sales_col or profit_col
                    if date_col and metric_col:
                        tmp = df.copy()
                        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce", utc=False)
                        tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
                        tmp = tmp.dropna(subset=[date_col, metric_col])
                        if tmp.empty:
                            raise ValueError("No usable rows for Global Superstore expert fallback")
                        tmp["date_bucket"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
                        chart_df = (
                            tmp.groupby("date_bucket", as_index=False)[metric_col]
                            .sum()
                            .rename(columns={"date_bucket": "date", metric_col: "value"})
                            .sort_values("date")
                        )
                        spec = ChartSpec(
                            chart_type=chosen_chart,
                            x_col="date",
                            y_col="value",
                            title=f"Global Superstore ({chosen_chart.title()})",
                        )
                    else:
                        fallback = infer_generic_fallback_spec(df, title=f"{dataset.name} — Overview")
                        if fallback is None: #If no numeric metric exists, use counts per month as a fallback
                            raise ValueError("No suitable columns found to build a chart.")
                        base_spec, chart_df = fallback
                        spec = ChartSpec(
                            chart_type=chosen_chart,
                            x_col=base_spec.x_col,
                            y_col=base_spec.y_col,
                            title=f"{dataset.name} ({chosen_chart.title()})",
                        )
            chart_html = make_chart_html(chart_df, spec)
            chart_json = make_chart_json(chart_df, spec)
            chart_explanation = build_chart_explanation_dict(
                plot_df=chart_df,
                chart_type=spec.chart_type,
                x_col=spec.x_col,
                y_col=spec.y_col,
            )
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": chart_explanation,
                    "spec": spec,
                    "chart_html": chart_html,
                    "chart_json": chart_json,
                    "error": None,
                    "mode": mode,
                    "chart_type": chosen_chart,
                    "columns": columns,
                    "selected_x_col": spec.x_col,
                    "selected_y_col": spec.y_col,
                },
            )
        except Exception as e:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "chart_explanation": None,
                    "error": f"Global Superstore visualisation failed: {e}",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
            )
    base_spec = infer_time_series_spec(df, title=f"{dataset.name} — Time Series")
    plot_df = df
    if base_spec is None:
        fallback = infer_generic_fallback_spec(df, title=f"{dataset.name} — Overview")
        if fallback is None:
            return templates.TemplateResponse(
                "dataset_visualise.html",
                {
                    "request": request,
                    "dataset": dataset,
                    "all_datasets": all_datasets,
                    "storyboards": storyboards,
                    "error": "No suitable columns found to build a chart.",
                    "mode": mode,
                    "chart_type": chart_type,
                    "columns": list(df.columns),
                    "selected_x_col": x_col,
                    "selected_y_col": y_col,
                },
            )
        base_spec, plot_df = fallback
    try: #Beginner mode uses the recommended chart, while expert mode uses the chart type selected in the UI.
        chosen_chart = base_spec.chart_type if mode == "beginner" else chart_type
        columns = list(df.columns) #If expert mode and user provided valid columns, override the automatic selection.
        if mode == "expert" and x_col and y_col and x_col in df.columns and y_col in df.columns:
            chart_df = prepare_expert_chart_df(df, x_col, y_col, chosen_chart)
            spec = ChartSpec(
                chart_type=chosen_chart,
                x_col=x_col,
                y_col=y_col,
                title=f"{dataset.name} ({chosen_chart.title()})",
            )
        else:
            chart_df = plot_df
            spec = ChartSpec(
                chart_type=chosen_chart,
                x_col=base_spec.x_col,
                y_col=base_spec.y_col,
                title=f"{base_spec.title} ({chosen_chart.title()})",
            )
        chart_html = make_chart_html(chart_df, spec)
        chart_json = make_chart_json(chart_df, spec)
        chart_explanation = build_chart_explanation_dict(
            plot_df=chart_df,
            chart_type=spec.chart_type,
            x_col=spec.x_col,
            y_col=spec.y_col,
        )
        return templates.TemplateResponse(
            "dataset_visualise.html",
            {
                "request": request,
                "dataset": dataset,
                "all_datasets": all_datasets,
                "storyboards": storyboards,
                "chart_explanation": chart_explanation,
                "spec": spec,
                "chart_html": chart_html,
                "chart_json": chart_json,
                "error": None,
                "mode": mode,
                "chart_type": chosen_chart,
                "columns": columns,
                "selected_x_col": spec.x_col,
                "selected_y_col": spec.y_col,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "dataset_visualise.html",
            {
                "request": request,
                "dataset": dataset,
                "all_datasets": all_datasets,
                "storyboards": storyboards,
                "chart_explanation": None,
                "error": f"Chart rendering failed: {e}",
                "mode": mode,
                "chart_type": chart_type,
                "columns": list(df.columns),
                "selected_x_col": x_col,
                "selected_y_col": y_col,
            },
    )

@app.get("/datasets/upload") #Upload Page: Upload form only, actual logic is done by post route below this
def dataset_upload_page(request: Request):
    return templates.TemplateResponse(
        "dataset_upload.html",
        {
            "request": request,
            "gdpr_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/",
            "error": None,
            "success": None,
        },
    )
@app.post("/datasets/upload") #Accepts CSV file, store on disk and register in MySQL
async def dataset_upload_post( 
    request: Request,
    file: UploadFile = File(...),
    consent: str | None = Form(None),
    db: Session = Depends(get_db),
): #Basic UK GDPR, user must tick consent box before uploading
    if consent != "yes": #Checks user has accepted the notice
        return templates.TemplateResponse(
            "dataset_upload.html",
            {
                "request": request,
                "gdpr_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/",
                "error": "You must confirm you have read the UK GDPR notice to upload a dataset.",
                "success": None,
            },
            status_code=400,
        )
    if not file or not file.filename: #Ensures that the file is there
        return templates.TemplateResponse(
            "dataset_upload.html",
            {
                "request": request,
                "gdpr_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/",
                "error": "No file selected.",
                "success": None,
            },
            status_code=400,
        )
    ext = Path(file.filename).suffix.lower() #Only accept CSV so the rest of the pipeline stays predictable
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "dataset_upload.html",
            {
                "request": request,
                "gdpr_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/",
                "error": "Only CSV files are allowed.",
                "success": None,
            },
            status_code=400,
        )
    data = _read_upload_bytes(file) #Reads uploads and adds a size limit
    _basic_csv_validation(data) #CSV must have minimum 1 row and 1 column
    safe_stem = _safe_stem(file.filename) #Safe filename
    unique_prefix = uuid.uuid4().hex
    stored_filename = f"{unique_prefix}_{safe_stem}.csv"
    stored_path = UPLOAD_DIR / stored_filename
    try: #Saves file to backend/uploads
        stored_path.write_bytes(data)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    base_name = safe_stem.replace("_", " ").title() #User friendly dataset for UI and ensures its unique in mysql
    dataset_name = _make_unique_dataset_name(db, base_name)
    rel_path = (Path("uploads") / stored_filename).as_posix()
    ds = Dataset( #Insert uplaoded dataset into same dataset table as other datasets
        name=dataset_name,
        domain="finance",
        source="User upload",
        description="User-uploaded dataset",
        file_path=rel_path,
        original_filename=file.filename,
        is_user_uploaded=True,
    )
    try:
        db.add(ds)
        db.commit()
        db.refresh(ds)
    except Exception: #If DB insert fails, remove saved file so no unused files left behind
        try:
            stored_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return templates.TemplateResponse( #Shows same upload page again 
        "dataset_upload.html",
        {
            "request": request,
            "gdpr_url": "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/",
            "error": None,
            "success": f"Upload successful. Dataset created with id={ds.id}.",
        },
        status_code=200,
    )

@app.get("/api/linking/suggestions") #For potential links in between the datasets
def api_linking_suggestions( #Logic is rule-based rather than ML-based
    dataset_a_id: int = Query(..., ge=1),
    dataset_b_id: int = Query(..., ge=1),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
): #Validate that both datasets exist in the database
    a = get_dataset_by_id(db, dataset_a_id)
    b = get_dataset_by_id(db, dataset_b_id)
    if not a or not b:
        raise HTTPException(
            status_code=404,
            detail="One or both datasets not found"
        ) #Prevent comparing a dataset with itself, it avoids unnecessary computation and simplifies UI handling.
    if a.id == b.id:
        return {
            "dataset_a_id": a.id,
            "dataset_b_id": b.id,
            "shared_exact": [],
            "suggestions": []
        }
    try: #Load datasets into memory as pandas DataFrames, reused existing loader for consistency for curated and uploaded datasets
        df_a = load_dataset_df(a.name, a.file_path)
        df_b = load_dataset_df(b.name, b.file_path)
    except Exception as e: #Prevents internal tracebacks from being exposed and ensures predictable client behaviour.
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load datasets for linking: {e}"
        )
    shared = shared_columns_exact(df_a, df_b) #Apply linking heuristics 
    suggestions = suggest_links(df_a, df_b, top_k=top_k) #overlap scoring 
    return { #Return structured JSON, Keeping this transformation outside UI maintains clean layering.
        "dataset_a_id": a.id,
        "dataset_b_id": b.id,
        "shared_exact": shared,
        "suggestions": suggestions_to_dict(suggestions),
    }

@app.get("/api/insights/generate") #Insights API to generate insights from two linked datasets
def api_generate_insights(
    dataset_a_id: int = Query(..., ge=1),
    dataset_b_id: int = Query(..., ge=1),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):#Validates datasets exist in MySQL so the endpoint fails fast with a controlled message
    a = get_dataset_by_id(db, dataset_a_id)
    b = get_dataset_by_id(db, dataset_b_id)
    if not a or not b:
        raise HTTPException(status_code=404, detail="One or both datasets not found")
    if a.id == b.id: #Avoids self-comparison to keep outputs meaningful and reduce unnecessary work
        return {
            "dataset_a_id": a.id,
            "dataset_b_id": b.id,
            "links": {"shared_exact": [], "suggestions": []},
            "insights": [],
        }
    try: #Loads both datasets using the same loader as the visualisation layer for consistency
        df_a = load_dataset_df(a.name, a.file_path)
        df_b = load_dataset_df(b.name, b.file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load datasets for insight generation: {e}")
    shared = shared_columns_exact(df_a, df_b) #Computes links first, because insights are derived from the linking output
    sugs = suggest_links(df_a, df_b, top_k=top_k)
    sugs_dict = suggestions_to_dict(sugs)
    insights = generate_insights(
        df_a=df_a,
        df_b=df_b,
        shared_exact=shared,
        suggestions=sugs_dict,
        dataset_a_name=a.name,
        dataset_b_name=b.name,
    )
    return {
        "dataset_a_id": a.id,
        "dataset_b_id": b.id,
        "links": {"shared_exact": shared, "suggestions": sugs_dict},
        "insights": insights,
    }

@app.post("/storyboard/save") #Route to save a visualisation into a storyboard
def save_to_storyboard(
    visualisation_id: int = Form(...),
    storyboard_id: int | None = Form(None),
    new_storyboard_title: str | None = Form(None),
    caption: str | None = Form(None),
    chart_html: str | None = Form(None),
    chart_json: str | None = Form(None),
    db: Session = Depends(get_db)
):
    print("chart_json received:", chart_json is not None)
    print("chart_json length:", len(chart_json) if chart_json else 0)
    storyboard = None
    clean_new_title = (new_storyboard_title or "").strip()
    if clean_new_title: #create a new storyboard if user entered a title
        storyboard = Storyboard(title=clean_new_title)
        db.add(storyboard)
        db.commit()
        db.refresh(storyboard)
    elif storyboard_id is not None: #otherwise use selected existing storyboard
        storyboard = (
            db.query(Storyboard)
            .filter(Storyboard.id == storyboard_id)
            .first()
        )
        if not storyboard:
            raise HTTPException(status_code=404, detail="Storyboard not found")
    else: #final fallback if neither was supplied
        storyboard = Storyboard(title="Untitled Storyboard")
        db.add(storyboard)
        db.commit()
        db.refresh(storyboard)
    max_order = (
        db.query(func.max(StoryboardSlide.order_index))
        .filter(StoryboardSlide.storyboard_id == storyboard.id)
        .scalar()
    )
    next_order = 0 if max_order is None else max_order + 1
    slide = StoryboardSlide(
        storyboard_id=storyboard.id,
        visualisation_id=None,
        caption=caption,
        chart_html=chart_html,
        chart_json=chart_json,
        user_notes="",
        order_index=next_order
    )
    db.add(slide)
    db.commit()
    db.refresh(slide)
    print("saved slide id:", slide.id)
    print("saved slide chart_json is not None:", slide.chart_json is not None)
    print("saved slide chart_json length:", len(slide.chart_json) if slide.chart_json else 0)
    return {
        "status": "saved",
        "slide_id": slide.id,
        "storyboard_id": storyboard.id
    }

@app.post("/storyboards/create") #Create empty storyboard and redirects to it too
def create_storyboard(
    title: str = Form(...),
    db: Session = Depends(get_db)
):
    clean_title = title.strip()
    if not clean_title:
        clean_title = "Untitled Storyboard"
    storyboard = Storyboard(title=clean_title)
    db.add(storyboard)
    db.commit()
    db.refresh(storyboard)
    return RedirectResponse(
        url=f"/storyboards/{storyboard.id}",
        status_code=303
    )

@app.post("/storyboards/{storyboard_id}/delete") #Removes the storyboard from the menu
def delete_storyboard(
    storyboard_id: int,
    db: Session = Depends(get_db)
):
    storyboard = (
        db.query(Storyboard)
        .filter(Storyboard.id == storyboard_id)
        .first()
    )
    if not storyboard:
        raise HTTPException(status_code=404, detail="Storyboard not found")

    db.delete(storyboard)
    db.commit()
    return RedirectResponse(
        url="/storyboards",
        status_code=303
    )

@app.get("/storyboards/{storyboard_id}") #Shows 1 storyboard and its slides, also edit the slide order
def view_storyboard(
    storyboard_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    storyboard = (
        db.query(Storyboard)
        .filter(Storyboard.id == storyboard_id)
        .first()
    )
    if not storyboard:
        raise HTTPException(status_code=404, detail="Storyboard not found")
    slides = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.storyboard_id == storyboard_id)
        .order_by(StoryboardSlide.order_index.asc())
        .all()
    )
    return templates.TemplateResponse(
        "storyboard.html",
        {
            "request": request,
            "storyboard": storyboard,
            "slides": slides,
        }
    )

@app.post("/storyboards/{storyboard_id}/slides/add") #User can add slides to storyboard for notes
def add_storyboard_slide(storyboard_id: int, db: Session = Depends(get_db)):
    storyboard = db.query(Storyboard).filter(
        Storyboard.id == storyboard_id
    ).first()
    if not storyboard:
        raise HTTPException(status_code=404, detail="Storyboard not found")
    next_order = len(storyboard.slides) #Determine next slide order
    slide = StoryboardSlide( #Create blank slide
        storyboard_id=storyboard_id,
        visualisation_id=None,
        chart_html="",
        caption="",
        user_notes="",
        order_index=next_order
    )
    db.add(slide)
    db.commit()
    return RedirectResponse(
        url=f"/storyboards/{storyboard_id}",
        status_code=303
    )

@app.post("/slides/{slide_id}/notes") #Each slide has the option to store speaker notes/explanation
def update_slide_notes(
    slide_id: int,
    user_notes: str = Form(""),
    db: Session = Depends(get_db)
):
    slide = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.id == slide_id)
        .first()
    )
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    slide.user_notes = user_notes
    db.commit()
    return RedirectResponse(
        url=f"/storyboards/{slide.storyboard_id}",
        status_code=303
    )

@app.post("/slides/{slide_id}/delete") #After slide is deleted, re-number the slides so order remains clean
def delete_slide(slide_id: int, db: Session = Depends(get_db)):
    slide = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.id == slide_id)
        .first()
    )
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    storyboard_id = slide.storyboard_id
    db.delete(slide)
    db.commit()
    remaining_slides = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.storyboard_id == storyboard_id)
        .order_by(StoryboardSlide.order_index.asc(), StoryboardSlide.id.asc())
        .all()
    )
    for index, s in enumerate(remaining_slides):
        s.order_index = index
    db.commit()
    return RedirectResponse(
        url=f"/storyboards/{storyboard_id}",
        status_code=303
    )

@app.post("/slides/{slide_id}/move") #Move slide up/down to reorder them
def move_slide(
    slide_id: int,
    direction: str = Form(...),
    db: Session = Depends(get_db)
):
    slide = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.id == slide_id)
        .first()
    )
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    storyboard_id = slide.storyboard_id
    slides = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.storyboard_id == storyboard_id)
        .order_by(StoryboardSlide.order_index.asc())
        .all()
    )
    current_index = None
    for i, s in enumerate(slides):
        if s.id == slide.id:
            current_index = i
            break
    if current_index is None:
        raise HTTPException(status_code=404, detail="Slide not found in storyboard")
    if direction == "up" and current_index > 0:
        other_slide = slides[current_index - 1]
    elif direction == "down" and current_index < len(slides) - 1:
        other_slide = slides[current_index + 1]
    else:
        return RedirectResponse(
            url=f"/storyboards/{storyboard_id}",
            status_code=303
        )
    slide.order_index, other_slide.order_index = other_slide.order_index, slide.order_index
    db.commit()
    return RedirectResponse(
        url=f"/storyboards/{storyboard_id}",
        status_code=303
    )

def extract_plotly_figure_from_html(chart_html: str): #Rebuild Plotly from a stored HTML, used as fallback if chart_json missing
    if not chart_html:
        return None
    try:
        match = re.search(r"Plotly\.newPlot\((.*?)\);", chart_html, re.DOTALL)
        if not match:
            return None
        content = match.group(1)
        parts = content.split(",", 2)
        if len(parts) < 3:
            return None
        data_json = parts[1].strip()
        layout_json = parts[2].rsplit("}", 1)[0] + "}"
        data = json.loads(data_json)
        layout = json.loads(layout_json)
        fig = go.Figure(data=data, layout=layout)
        return fig
    except Exception:
        return None

@app.get("/storyboards/{storyboard_id}/export-pdf")
def export_storyboard_pdf(
    storyboard_id: int,
    db: Session = Depends(get_db)
):
    storyboard = (
        db.query(Storyboard)
        .filter(Storyboard.id == storyboard_id)
        .first()
    )

    if not storyboard:
        raise HTTPException(status_code=404, detail="Storyboard not found")

    slides = (
        db.query(StoryboardSlide)
        .filter(StoryboardSlide.storyboard_id == storyboard_id)
        .order_by(StoryboardSlide.order_index.asc())
        .all()
    )

    # Create a temporary PDF file safely
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = tmp.name
    tmp.close()

    pdf = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    left_margin = 20 * mm
    right_margin = 20 * mm
    top_margin = 20 * mm
    bottom_margin = 20 * mm
    usable_width = page_width - left_margin - right_margin

    def draw_wrapped_text(
        text: str,
        x: float,
        y: float,
        max_width: float,
        line_height: float = 14
    ):
        if not text:
            return y

        words = text.split()
        current_line = ""

        for word in words:
            test_line = word if not current_line else f"{current_line} {word}"
            if pdf.stringWidth(test_line, "Helvetica", 10) <= max_width:
                current_line = test_line
            else:
                pdf.drawString(x, y, current_line)
                y -= line_height
                current_line = word

        if current_line:
            pdf.drawString(x, y, current_line)
            y -= line_height

        return y

    for index, slide in enumerate(slides, start=1):
        y = page_height - top_margin

        # Header
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(left_margin, y, storyboard.title)
        y -= 10 * mm

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left_margin, y, f"Slide {index}")
        y -= 8 * mm

        # Chart placeholder box
        chart_height = 60 * mm
        pdf.setFont("Helvetica", 10)
        pdf.rect(left_margin, y - chart_height, usable_width, chart_height)

        if slide.chart_json and slide.chart_json.strip():
            try:
                fig = go.Figure(json.loads(slide.chart_json))

                img_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img_path = img_tmp.name
                img_tmp.close()

                fig.write_image(img_path, width=800, height=400)
                pdf.drawImage(
                    img_path,
                    left_margin,
                    y - chart_height,
                    width=usable_width,
                    height=chart_height,
                    preserveAspectRatio=True,
                    mask="auto"
                )
            except Exception as e:
                print(f"PDF export chart render failed for slide {slide.id}: {e}")
                pdf.drawString(
                    left_margin + 5 * mm,
                    y - 10 * mm,
                    "Failed to render the chart image"
                )
        else:
            pdf.drawString(
                left_margin + 5 * mm,
                y - 10 * mm,
                "There is no chart on this slide"
            )

        y -= 70 * mm


        # Notes section
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(left_margin, y, "Notes")
        y -= 6 * mm

        pdf.setFont("Helvetica", 10)
        notes_text = slide.user_notes.strip() if slide.user_notes else "No notes provided."
        y = draw_wrapped_text(notes_text, left_margin, y, usable_width)

        # Footer
        pdf.setFont("Helvetica", 9)
        pdf.drawRightString(
            page_width - right_margin,
            bottom_margin / 2,
            f"Page {index}"
        )

        pdf.showPage()

    pdf.save()

    safe_title = "".join(
        ch if ch.isalnum() or ch in (" ", "-", "_") else "_"
        for ch in storyboard.title
    ).strip()

    if not safe_title:
        safe_title = "storyboard"

    filename = f"{safe_title}.pdf"

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename
    )

@app.get("/storyboards/{storyboard_id}/view", response_class=HTMLResponse)
def view_storyboard_public(
    storyboard_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Public read-only view for a storyboard.
    It only displays saved charts and notes.
    """
    storyboard = (
        db.query(Storyboard)
        .options(selectinload(Storyboard.slides))
        .filter(Storyboard.id == storyboard_id)
        .first()
    )

    if storyboard is None:
        raise HTTPException(status_code=404, detail="Storyboard not found")

    ordered_slides = sorted(
        storyboard.slides,
        key=lambda slide: (
            slide.order_index if slide.order_index is not None else 0,
            slide.id,
        ),
    )

    return templates.TemplateResponse(
        "storyboard_public_view.html",
        {
            "request": request,
            "storyboard": storyboard,
            "slides": ordered_slides,
        },
    )

@app.get("/storyboards")
def list_storyboards(
    request: Request,
    db: Session = Depends(get_db)
):
    storyboards = (
        db.query(Storyboard)
        .order_by(Storyboard.created_at.desc(), Storyboard.id.desc())
        .all()
    )

    return templates.TemplateResponse(
        "storyboards.html",
        {
            "request": request,
            "storyboards": storyboards,
        }
    )

@app.post("/storyboards/create")
def create_storyboard(
    title: str = Form(...),
    db: Session = Depends(get_db)
):
    clean_title = title.strip()

    if not clean_title:
        clean_title = "Untitled Storyboard"

    storyboard = Storyboard(title=clean_title)
    db.add(storyboard)
    db.commit()
    db.refresh(storyboard)

    return RedirectResponse(
        url=f"/storyboards/{storyboard.id}",
        status_code=303
    )

templates = Jinja2Templates(directory="app/templates")

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    # temporary dummy logic (replace later with DB check)
    if email == "test@test.com" and password == "123":
        request.session["user"] = email
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": "Invalid email or password"
        }
    )

@app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    response.delete_cookie("session")
    return response

@app.get("/feedback", response_class=HTMLResponse)
def feedback_page(request: Request):
    return templates.TemplateResponse("feedback.html", {"request": request})

#MIDDLEWARE: This controls if browser can call bacend from otherplaces
#For testing only as in production I would resrtrict the allow_origins to my actual domain rather than *
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#HEALTH CHECK: Endpoint to confirm backend is running for deploying and debugging
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug/domains")
def debug_domains(db: Session = Depends(get_db)):
    rows = db.execute(text("SELECT DISTINCT domain FROM datasets")).fetchall()
    return {"domains": [r[0] for r in rows]}

#API: DATASETS: JSON endpoints (API layer), separate from the HTML pages
#UI pages use Jinja templates, but the API is useful for testing and future extension
@app.get("/datasets", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)): #Returns datasets from mysql as JSON
    return db.query(Dataset).order_by(Dataset.id).all()

@app.get("/datasets/{dataset_id}", response_model=DatasetOut) #Returns datasets metadata by id again for debugging
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = get_dataset_by_id(db, dataset_id) #JSON dataset preview for testing
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.get("/datasets/{dataset_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(
    dataset_id: int,
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    dataset = get_dataset_by_id(db, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    rows = get_dataset_preview_rows(db, dataset_id, limit)
    return {"dataset_id": dataset_id, "limit": limit, "rows": rows}

#API: INSIGHTS: generate or retrieve text insights for a visualisation
@app.get("/insights", response_model=InsightOut)
def get_insights(visualisation_id: int = Query(..., ge=1), db: Session = Depends(get_db)):
    insights = generate_insights(db, visualisation_id=visualisation_id, limit=500)
    if insights is None:
        raise HTTPException(status_code=404, detail="Visualisation not found")
    return insights

#API: VISUALISATIONS: store and retrieve visualisation configs so user can recreate the same chart later without re-selecting options
@app.post("/visualisations", response_model=VisualisationOut)
def create_visualisation(payload: VisualisationCreate, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, payload.dataset_id) #Validate dataset exists before saving a visualisation for it
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    v = Visualisation( #Store chart settings as JSON so it supports different chart types without changing the schema
        dataset_id=payload.dataset_id,
        chart_type=payload.chart_type,
        title=payload.title,
        spec_json=json.dumps(payload.spec),
    )

    db.add(v)
    db.commit()
    db.refresh(v)

    return VisualisationOut(
        id=v.id,
        dataset_id=v.dataset_id,
        chart_type=v.chart_type,
        title=v.title,
        spec=json.loads(v.spec_json),
        created_at=v.created_at,
    )

#Returns all saved visualisations for one dataset (newest first)
@app.get("/visualisations", response_model=list[VisualisationOut])
def list_visualisations(dataset_id: int, db: Session = Depends(get_db)):
    rows = (
        db.query(Visualisation)
        .filter(Visualisation.dataset_id == dataset_id)
        .order_by(Visualisation.created_at.desc())
        .all()
    )
    #Convert each row into the response model
    return [
        VisualisationOut(
            id=r.id,
            dataset_id=r.dataset_id,
            chart_type=r.chart_type,
            title=r.title,
            spec=json.loads(r.spec_json),
            created_at=r.created_at,
        )
        for r in rows
    ]