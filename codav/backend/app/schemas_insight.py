from pydantic import BaseModel
from typing import Dict


#Schema used for insight responses returned by the API
#Represents lightweight automatically generated statistics for a visualisation
#Designed for MVP so structure stays simple and easy to extend later
class InsightOut(BaseModel):

    #Links insights back to the visualisation they were generated from
    visualisation_id: int

    #Basic dataset summary such as number of rows and columns
    #Example {"rows":120,"columns":5}
    summary: Dict[str, int]

    #Dictionary of missing value counts per column
    #Example {"sales":0,"month":0}
    missing_values: Dict[str, int]