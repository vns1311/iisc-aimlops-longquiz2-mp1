import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from datetime import datetime

from obesity_model.config.core import config


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    validated_data = input_df[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    Gender: Optional[str]
    Age: Optional[float]
    Height: Optional[float]
    Weight: Optional[float]
    family_history_with_overweight: Optional[str]
    FAVC: Optional[str]
    FCVC: Optional[float]
    NCP: Optional[float]
    CAEC: Optional[str]
    SMOKE: Optional[str]
    CH2O: Optional[float]
    SCC: Optional[str]
    FAF: Optional[float]
    TUE: Optional[float]
    CALC: Optional[str]
    MTRANS: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
