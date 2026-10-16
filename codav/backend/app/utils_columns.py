from __future__ import annotations


#Converts raw column names into cleaner labels for display in the UI
#Acts as a small safety layer when datasets contain messy headers like Unnamed columns
def display_label(col_name: str) -> str:

    #Fallback label if column name is missing or null
    if col_name is None:
        return "Unknown"

    c = str(col_name).strip()

    #Some datasets include placeholder headers like Unnamed: 0
    #These are mapped to a more meaningful default label
    if c == "" or c.lower().startswith("unnamed"):
        return "Country"

    #Return cleaned column name for display
    return c