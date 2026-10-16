from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class Insight:
    insight_type: str
    title: str
    summary: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalise_col_name(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch == "_")


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _find_year_column(df: pd.DataFrame) -> Optional[str]:
    #Common names used across datasets
    candidates = ["year", "date", "timestamp"]

    for c in candidates:
        if c in df.columns:
            return c

    #Fallback: detect numeric column that looks like a year
    for c in df.columns:
        if df[c].dtype.kind in {"i", "u"}:
            sample = df[c].dropna().head(50)
            if not sample.empty and sample.between(1800, 2200).mean() > 0.9:
                return c

    return None


def _infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols: List[str] = []

    ignore_patterns = ["id", "index", "row"]

    for c in df.columns:
        name = c.lower()

        #Skip identifier-style columns
        if any(p in name for p in ignore_patterns):
            continue

        s = _safe_numeric_series(df[c])

        #Keep columns with strong numeric signal
        if s.notna().mean() >= 0.6:
            numeric_cols.append(c)

    return numeric_cols


def _is_missing_value(value: Any) -> bool:
    if pd.isna(value):
        return True

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in {"", "nan", "null", "none", "n/a", "na"}:
            return True

    return False


def _generate_data_quality_insights(
    df: pd.DataFrame,
    dataset_name: str,
) -> List[Insight]:

    insights: List[Insight] = []

    if df.empty:
        return insights

    missing_stats: List[Dict[str, Any]] = []

    affected_columns = 0
    complete_columns = 0

    for col in df.columns:

        series = df[col]

        total = len(series)

        missing = int(series.apply(_is_missing_value).sum())

        pct = (missing / total) * 100 if total else 0

        missing_stats.append(
            {
                "column": col,
                "missing_count": missing,
                "total": total,
                "missing_pct": round(pct, 1),
            }
        )

        if missing == 0:
            complete_columns += 1
        else:
            affected_columns += 1

    if complete_columns == len(df.columns):

        insights.append(
            Insight(
                insight_type="data_quality",
                title=f"{dataset_name}: complete preview data",
                summary=(
                    f"All previewed columns in {dataset_name} appear complete "
                    f"with no missing values detected."
                ),
                evidence={
                    "dataset": dataset_name,
                    "columns": len(df.columns),
                },
            )
        )

        return insights

    insights.append(
        Insight(
            insight_type="data_quality",
            title=f"{dataset_name}: missing data detected",
            summary=(
                f"{affected_columns} column(s) contain missing values "
                f"which may affect analysis or visualisation clarity."
            ),
            evidence={
                "dataset": dataset_name,
                "affected_columns": affected_columns,
                "column_missingness": missing_stats[:20],
            },
        )
    )

    return insights


def _generate_numeric_distribution_insights(
    df: pd.DataFrame,
    dataset_name: str,
) -> List[Insight]:

    insights: List[Insight] = []

    numeric_cols = _infer_numeric_columns(df)

    if not numeric_cols:
        return insights

    col = numeric_cols[0]

    series = _safe_numeric_series(df[col]).dropna()

    if series.empty:
        return insights

    min_val = float(series.min())
    max_val = float(series.max())
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std())

    summary = (
        f"The values in '{col}' range from {min_val:,.2f} to {max_val:,.2f} "
        f"with an average of approximately {mean_val:,.2f}."
    )

    if std_val / mean_val < 0.1:
        summary += " The values appear relatively stable."
    else:
        summary += " The values show noticeable variation."

    insights.append(
        Insight(
            insight_type="numeric_distribution",
            title=f"{dataset_name}: value distribution",
            summary=summary,
            evidence={
                "dataset": dataset_name,
                "column": col,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
            },
        )
    )

    return insights


def _generate_time_series_trend_insights(
    df: pd.DataFrame,
    dataset_name: str,
) -> List[Insight]:

    insights: List[Insight] = []

    year_col = _find_year_column(df)
    numeric_cols = _infer_numeric_columns(df)

    if not year_col or not numeric_cols:
        return insights

    value_col = numeric_cols[0]

    time_series = pd.to_datetime(df[year_col], errors="coerce")

    values = _safe_numeric_series(df[value_col])

    valid = (~time_series.isna()) & (~values.isna())

    if valid.sum() < 5:
        return insights

    df_sorted = pd.DataFrame(
        {
            "time": time_series[valid],
            "value": values[valid],
        }
    ).sort_values("time")

    start_val = float(df_sorted["value"].iloc[0])
    end_val = float(df_sorted["value"].iloc[-1])

    change = end_val - start_val

    pct_change = (change / start_val) * 100 if start_val else 0

    if abs(pct_change) < 5:
        trend = "remains relatively stable"
    elif pct_change > 0:
        trend = "shows an increasing trend"
    else:
        trend = "shows a decreasing trend"

    insights.append(
        Insight(
            insight_type="time_trend",
            title=f"{dataset_name}: time-series trend",
            summary=(
                f"The values of '{value_col}' {trend} over the observed period. "
                f"The series starts around {start_val:,.2f} and ends around {end_val:,.2f}."
            ),
            evidence={
                "dataset": dataset_name,
                "time_column": year_col,
                "value_column": value_col,
                "start": start_val,
                "end": end_val,
                "pct_change": pct_change,
            },
        )
    )

    return insights


def generate_insights(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    shared_exact: List[Dict[str, str]],
    suggestions: List[Dict[str, Any]],
    dataset_a_name: str,
    dataset_b_name: str,
    max_items: int = 12,
) -> List[Dict[str, Any]]:

    insights: List[Insight] = []

    #Schema overlap
    if shared_exact:
        insights.append(
            Insight(
                insight_type="schema_overlap",
                title="Shared attributes detected",
                summary=(
                    f"{dataset_a_name} and {dataset_b_name} share "
                    f"{len(shared_exact)} exact column match(es)."
                ),
                evidence={"shared_exact": shared_exact[:20]},
            )
        )
    else:
        insights.append(
            Insight(
                insight_type="schema_overlap",
                title="No exact schema matches",
                summary=(
                    f"No exact column matches were detected between "
                    f"{dataset_a_name} and {dataset_b_name}."
                ),
                evidence={},
            )
        )

    #Suggested links
    if suggestions:
        insights.append(
            Insight(
                insight_type="suggested_links",
                title="Suggested column relationships",
                summary="Potential links between datasets were detected using heuristic matching.",
                evidence={"suggestions": suggestions[:5]},
            )
        )

    #Dataset size
    insights.append(
        Insight(
            insight_type="dataset_shape",
            title="Dataset size comparison",
            summary=(
                f"{dataset_a_name} contains {len(df_a):,} rows and {df_a.shape[1]} columns, "
                f"while {dataset_b_name} contains {len(df_b):,} rows and {df_b.shape[1]} columns."
            ),
            evidence={
                "dataset_a_rows": len(df_a),
                "dataset_b_rows": len(df_b),
            },
        )
    )

    insights.extend(_generate_data_quality_insights(df_a, dataset_a_name))
    insights.extend(_generate_data_quality_insights(df_b, dataset_b_name))

    insights.extend(_generate_numeric_distribution_insights(df_a, dataset_a_name))
    insights.extend(_generate_numeric_distribution_insights(df_b, dataset_b_name))

    insights.extend(_generate_time_series_trend_insights(df_a, dataset_a_name))
    insights.extend(_generate_time_series_trend_insights(df_b, dataset_b_name))

    return [i.to_dict() for i in insights[:max_items]]