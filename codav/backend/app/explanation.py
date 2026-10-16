#app/explanation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ExplanationResult:
    overview: str
    pattern: str
    interpretation: str
    caveat: str

    @property
    def full_text(self) -> str:
        return f"{self.overview} {self.pattern} {self.interpretation} {self.caveat}"


def safe_series_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    sample = series.dropna().astype(str).head(10)
    if sample.empty:
        return False

    converted = pd.to_datetime(sample, errors="coerce")
    return converted.notna().sum() >= max(1, len(sample) // 2)


def normalise_plot_df(plot_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    df = plot_df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=[x_col, y_col])

    df = df[[x_col, y_col]].copy()
    df = df.dropna(subset=[x_col, y_col])

    if is_datetime_like(df[x_col]):
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")

    df[y_col] = safe_series_to_numeric(df[y_col])
    df = df.dropna(subset=[y_col])

    return df


def describe_variation(y: pd.Series) -> str:
    if y.empty:
        return "the values are limited"

    y_range = y.max() - y.min()
    y_mean = y.mean()

    if y_mean == 0:
        ratio = 0
    else:
        ratio = abs(y_range / y_mean)

    if ratio < 0.15:
        return "the values remain fairly consistent"
    if ratio < 0.50:
        return "the values show some variation"
    return "the values vary noticeably"


def describe_trend(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if len(df) < 3:
        return "there is not enough data to identify a clear trend"

    working = df.copy()

    if is_datetime_like(working[x_col]):
        working = working.sort_values(x_col)
        working["_x_num"] = range(len(working))
    else:
        if pd.api.types.is_numeric_dtype(working[x_col]):
            working = working.sort_values(x_col)
            working["_x_num"] = pd.to_numeric(working[x_col], errors="coerce")
        else:
            return "there is no clear overall trend because the x-axis is categorical"

    working = working.dropna(subset=["_x_num", y_col])

    if len(working) < 3:
        return "there is not enough data to identify a clear trend"

    first = working[y_col].iloc[0]
    last = working[y_col].iloc[-1]

    if first == 0:
        pct_change = 0
    else:
        pct_change = (last - first) / abs(first)

    if pct_change > 0.10:
        return "there is a slight overall upward trend across the chart, although the pattern is not consistent"
    if pct_change < -0.10:
        return "there is a slight overall downward trend across the chart, although the pattern is not consistent"
    return "there is no strong upward or downward trend overall"


def describe_bar_pattern(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if df.empty:
        return "the chart does not contain enough data to describe a pattern"

    variation = describe_variation(df[y_col])

    if pd.api.types.is_numeric_dtype(df[x_col]) or is_datetime_like(df[x_col]):
        trend = describe_trend(df, x_col, y_col)
        return f"{variation}, and {trend}."
    return f"{variation} across the categories shown."


def describe_line_pattern(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if df.empty:
        return "the chart does not contain enough data to describe a pattern"

    variation = describe_variation(df[y_col])
    trend = describe_trend(df, x_col, y_col)
    return f"{variation}, and {trend}."


def describe_scatter_pattern(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if df.empty or len(df) < 3:
        return "there is not enough data to identify a clear relationship."

    x_num = pd.to_numeric(df[x_col], errors="coerce")
    y_num = pd.to_numeric(df[y_col], errors="coerce")
    valid = pd.DataFrame({x_col: x_num, y_col: y_num}).dropna()

    if len(valid) < 3:
        return "the points can be compared visually, but there is not enough numeric data to estimate a relationship."

    corr = valid[x_col].corr(valid[y_col])

    if pd.isna(corr):
        return "the points do not show a clear relationship."

    if corr > 0.60:
        return "the points show a fairly clear positive relationship, meaning higher values on one axis tend to be associated with higher values on the other."
    if corr < -0.60:
        return "the points show a fairly clear negative relationship, meaning higher values on one axis tend to be associated with lower values on the other."
    if abs(corr) >= 0.30:
        return "the points suggest a weak to moderate relationship, although it is not especially strong."
    return "the points appear fairly scattered, so there is no strong visible relationship."


def build_overview(chart_type: str, x_col: str, y_col: str) -> str:
    chart_type = (chart_type or "").lower()

    if chart_type == "line":
        return f"This line chart shows how {y_col} changes over {x_col}."
    if chart_type == "bar":
        return f"This bar chart shows how {y_col} varies across {x_col}."
    if chart_type == "scatter":
        return f"This scatter chart shows the relationship between {x_col} and {y_col}."
    return f"This chart shows {y_col} in relation to {x_col}."


def build_interpretation(chart_type: str, pattern_text: str) -> str:
    chart_type = (chart_type or "").lower()
    lower_pattern = pattern_text.lower()

    if chart_type == "line":
        if "upward trend" in lower_pattern:
            return "This suggests the measured values are generally increasing over the period shown."
        if "downward trend" in lower_pattern:
            return "This suggests the measured values are generally decreasing over the period shown."
        return "This suggests the measured values are relatively stable overall, even if there are smaller fluctuations."

    if chart_type == "bar":
        if "categories" in lower_pattern:
            return "This makes it easier to see which categories have higher or lower values."
        if "upward trend" in lower_pattern or "downward trend" in lower_pattern:
            return "This helps show how the values change across the order presented on the chart."
        return "This gives a simple comparison of differences in size between the bars."

    if chart_type == "scatter":
        if "positive relationship" in lower_pattern:
            return "This suggests the two variables may move in the same direction."
        if "negative relationship" in lower_pattern:
            return "This suggests the two variables may move in opposite directions."
        if "no strong visible relationship" in lower_pattern or "fairly scattered" in lower_pattern:
            return "This suggests one variable does not clearly predict the other from the chart alone."
        return "This gives a visual indication of whether the two variables may be related."

    return "This gives a simple summary of the main visual pattern in the chart."


def build_caveat(chart_type: str) -> str:
    chart_type = (chart_type or "").lower()

    if chart_type == "scatter":
        return "However, correlation does not necessarily mean causation, so this chart alone cannot prove that one variable causes changes in the other."
    if chart_type in {"line", "bar"}:
        return "However, this chart only shows the pattern in the displayed data and does not explain the underlying cause of that pattern."
    return "However, this chart should be interpreted with the wider context of the dataset in mind."


def generate_chart_explanation(
    plot_df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
) -> ExplanationResult:
    chart_type = (chart_type or "").lower().strip()
    cleaned_df = normalise_plot_df(plot_df, x_col, y_col)

    overview = build_overview(chart_type, x_col, y_col)

    if chart_type == "line":
        pattern = describe_line_pattern(cleaned_df, x_col, y_col)
    elif chart_type == "bar":
        pattern = describe_bar_pattern(cleaned_df, x_col, y_col)
    elif chart_type == "scatter":
        pattern = describe_scatter_pattern(cleaned_df, x_col, y_col)
    else:
        pattern = "The chart shows the main values visually, although a more specific pattern description is not available for this chart type."

    interpretation = build_interpretation(chart_type, pattern)
    caveat = build_caveat(chart_type)

    return ExplanationResult(
        overview=overview,
        pattern=pattern[:1].upper() + pattern[1:] if pattern else pattern,
        interpretation=interpretation,
        caveat=caveat,
    )

def build_chart_explanation_dict(
    plot_df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
) -> dict | None:
    try:
        explanation_result = generate_chart_explanation(
            plot_df=plot_df,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
        )
        return {
            "overview": explanation_result.overview,
            "pattern": explanation_result.pattern,
            "interpretation": explanation_result.interpretation,
            "caveat": explanation_result.caveat,
            "full_text": explanation_result.full_text,
        }
    except Exception as e:
        print("Chart explanation generation failed:", e)
        return None