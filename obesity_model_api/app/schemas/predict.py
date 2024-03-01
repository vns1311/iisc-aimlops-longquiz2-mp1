from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from obesity_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[str]]
    # predictions: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Gender": "Female",
                        "Age": 21.0,
                        "Height": 1.62,
                        "Weight": 64.0,
                        "family_history_with_overweight": "yes",
                        "FAVC": "no",
                        "FCVC": 2.0,
                        "NCP": 3.0,
                        "CAEC": "Sometimes",
                        "SMOKE": "no",
                        "CH2O": 2.0,
                        "SCC": "no",
                        "FAF": 0.0,
                        "TUE": 1.0,
                        "CALC": "no",
                        "MTRANS": "Public_Transportation"
                    }
                ]
            }
        }
