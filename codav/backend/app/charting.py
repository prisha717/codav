from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd

#Small contract used by the visualise page so the UI knows what chart to build
@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    x_col: str
    y_col: str
    title: str

#Detects year columns where headers are like 1960, 1961, 2020 etc
#Used when reshaping time-series datasets like World Bank or OECD
def find_year_columns(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    year_cols: list[str] = []
    for c in df.columns:
        s = str(c).strip()
        if s.isdigit() and len(s) == 4:
            year_cols.append(c)
    try:
        year_cols = sorted(year_cols, key=lambda x: int(str(x)))
    except Exception:
        pass
    return year_cols

#Helper to stop IDs being chosen as y-axis values
#IDs are numeric but meaningless for charts
def _looks_like_id_column(col_name: str) -> bool:
    n = col_name.strip().lower()
    bad_tokens = [
        "id", "uuid", "guid", "row id", "row_id", "index",
        "transactionid", "transaction id",
        "order id", "order_id",
        "customer id", "customer_id",
        "product id", "product_id",
        "code",
    ]
    return any(t in n for t in bad_tokens)

#Scores numeric columns so the system can guess a sensible y-axis
#Higher score means more likely to be chosen for beginner mode
def _score_numeric_y(col_name: str) -> int:
    name = col_name.strip().lower()
    if _looks_like_id_column(name):
        return -1000

    good = [
        "sales", "profit", "revenue", "amount", "total", "value",
        "price", "cost", "quantity", "qty", "units",
        "inflation", "gdp", "area",
    ]
    score = 0
    for g in good:
        if g in name:
            score += 50
    if "number" in name or "num" in name:
        score -= 5
    return score

#Tries to automatically choose x and y columns for a time-series chart
#Used by dataset_visualise_page when there is no dataset-specific override
def infer_time_series_spec(df: pd.DataFrame, title: str = "Time Series") -> Optional[ChartSpec]:
    if df is None or df.empty:
        return None

    #Find datetime columns first since they make the best x-axis
    datetime_candidates: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_candidates.append(col)

    #If no datetime dtype exists try parsing columns that look like dates
    if not datetime_candidates:
        for col in df.columns:
            name = str(col).strip().lower()
            if name in {"date", "order date", "timestamp", "time"} or "date" in name or "time" in name:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
                if parsed.notna().sum() >= max(3, int(0.5 * len(df))):
                    df[col] = parsed
                    datetime_candidates.append(col)

    #Fallback option is year-like numeric columns
    year_candidates: list[str] = []
    for col in df.columns:
        name = str(col).strip().lower()
        if name in {"year", "yr"}:
            year_candidates.append(col)
        elif pd.api.types.is_integer_dtype(df[col]):
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if not vals.empty:
                mn = int(vals.min())
                mx = int(vals.max())
                if mn >= 1900 and mx <= 2100:
                    year_candidates.append(col)

    #Find numeric columns for possible y-axis
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    x_col = datetime_candidates[0] if datetime_candidates else (year_candidates[0] if year_candidates else None)
    if not x_col or not numeric_cols:
        return None

    y_candidates = [c for c in numeric_cols if c != x_col]
    if not y_candidates:
        return None

    #Sort numeric columns using the scoring function
    y_candidates_sorted = sorted(y_candidates, key=lambda c: _score_numeric_y(str(c)), reverse=True)
    y_col = y_candidates_sorted[0]

    #Reject result if best column still looks like an ID
    if _looks_like_id_column(str(y_col)):
        return None

    return ChartSpec(chart_type="line", x_col=x_col, y_col=y_col, title=title)

#Generic fallback for datasets that are not time series
#Used when infer_time_series_spec returns None so beginner mode still renders a chart
def infer_generic_fallback_spec(df: pd.DataFrame, title: str = "Overview") -> tuple[ChartSpec, pd.DataFrame] | None:
    if df is None or df.empty:
        return None

    tmp = df.copy()

    # Try to coerce numeric columns that were read as strings
    numeric_cols: list[str] = []
    for c in tmp.columns:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            numeric_cols.append(c)
            continue
        coerced = pd.to_numeric(tmp[c], errors="coerce")
        if coerced.notna().sum() >= max(3, int(0.3 * len(tmp))):
            tmp[c] = coerced
            numeric_cols.append(c)

    numeric_cols = [c for c in numeric_cols if not _looks_like_id_column(str(c))]

    # Categorical columns for grouping
    cat_cols: list[str] = []
    for c in tmp.columns:
        if c in numeric_cols:
            continue
        if pd.api.types.is_bool_dtype(tmp[c]) or pd.api.types.is_object_dtype(tmp[c]) or pd.api.types.is_categorical_dtype(tmp[c]):
            cat_cols.append(c)

    def _pick_best_numeric(cols: list[str]) -> str | None:
        if not cols:
            return None
        scored = sorted(cols, key=lambda c: (_score_numeric_y(str(c)), tmp[c].notna().sum()), reverse=True)
        return scored[0]

    def _pick_best_category(cols: list[str]) -> str | None:
        if not cols:
            return None
        best = None
        best_score = -1
        for c in cols:
            non_null = tmp[c].dropna()
            if non_null.empty:
                continue
            uniq = non_null.nunique(dropna=True)
            if uniq < 2:
                continue
            if uniq > 50:
                continue
            score = non_null.notna().sum() - uniq
            if score > best_score:
                best = c
                best_score = score
        return best

    y_col = _pick_best_numeric(numeric_cols)
    x_cat = _pick_best_category(cat_cols)

    # Case 1: category + numeric -> bar chart
    if x_cat and y_col:
        g = tmp[[x_cat, y_col]].dropna()
        if not g.empty:
            out = g.groupby(x_cat, as_index=False)[y_col].mean()
            out = out.sort_values(by=y_col, ascending=False).head(15)
            out = out.rename(columns={x_cat: "category", y_col: "value"})
            spec = ChartSpec(chart_type="bar", x_col="category", y_col="value", title=title)
            return spec, out

    if len(numeric_cols) >= 2:
        x_num = numeric_cols[0]
        y_num = numeric_cols[1]

        out = tmp[[x_num, y_num]].dropna()
        if not out.empty:
            spec = ChartSpec(chart_type="scatter", x_col=x_num, y_col=y_num, title=title)
            return spec, out

    # Case 2: numeric only -> histogram
    if y_col and not x_cat:
        s = tmp[y_col].dropna()
        if not s.empty:
            bins = min(10, max(5, int(s.nunique() ** 0.5)))
            binned = pd.cut(s, bins=bins)
            out = binned.value_counts().sort_index().reset_index()
            out.columns = ["bin", "count"]
            out["bin"] = out["bin"].astype(str)
            spec = ChartSpec(chart_type="bar", x_col="bin", y_col="count", title=title)
            return spec, out

    # Case 3: category only -> counts
    if x_cat and not y_col:
        s = tmp[x_cat].dropna()
        if not s.empty:
            out = s.value_counts().head(15).reset_index()
            out.columns = ["category", "count"]
            spec = ChartSpec(chart_type="bar", x_col="category", y_col="count", title=title)
            return spec, out

    return None
def make_chart_html(df: pd.DataFrame, spec: ChartSpec) -> str:
    import plotly.express as px

    if df is None or df.empty:
        return "<p>No data available to plot.</p>"

    plot_df = df[[spec.x_col, spec.y_col]].dropna()
    if plot_df.empty:
        return "<p>No plottable rows after removing missing values.</p>"

    if spec.chart_type == "line":
        fig = px.line(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    elif spec.chart_type == "bar":
        fig = px.bar(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    elif spec.chart_type == "scatter":
        fig = px.scatter(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    else:
        return f"<p>Unsupported chart type: {spec.chart_type}</p>"

    return fig.to_html(full_html=False, include_plotlyjs="cdn")

def make_chart_json(df: pd.DataFrame, spec: ChartSpec) -> str | None:
    import plotly.express as px

    if df is None or df.empty:
        return None

    plot_df = df[[spec.x_col, spec.y_col]].dropna()
    if plot_df.empty:
        return None

    if spec.chart_type == "line":
        fig = px.line(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    elif spec.chart_type == "bar":
        fig = px.bar(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    elif spec.chart_type == "scatter":
        fig = px.scatter(plot_df, x=spec.x_col, y=spec.y_col, title=spec.title)
    else:
        return None

    return fig.to_json()