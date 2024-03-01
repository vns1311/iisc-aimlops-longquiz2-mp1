import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np
import json 
from obesity_model import __version__ as _version
from obesity_model.config.core import config
from obesity_model.pipeline import obesity_pipe
from obesity_model.processing.data_manager import load_pipeline
from obesity_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
obesity_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model"""

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    validated_data = validated_data.reindex(columns=config.model_config.features)

    results = {"predictions": None, "version": _version, "errors": errors}

    predictions = obesity_pipe.predict(validated_data)

    results = {"predictions": predictions.tolist(), "version": _version, "errors": errors}
    print(results)
    if not errors:
        predictions = obesity_pipe.predict(validated_data)
        results = {"predictions": predictions, "version": _version, "errors": errors}

    return results


if __name__ == "__main__":
    data_in = {
        "Gender": ["Female", "Female"],
        "Age": [21.0, 21.0],
        "Height": [1.62, 1.52],
        "Weight": [64.0, 56.0],
        "family_history_with_overweight": ["yes", "yes"],
        "FAVC": ["no", "no"],
        "FCVC": [2.0, 3.0],
        "NCP": [3.0, 3.0],
        "CAEC": ["Sometimes", "Sometimes"],
        "SMOKE": ["no", "yes"],
        "CH2O": [2.0, 3.0],
        "SCC": ["no", "yes"],
        "FAF": [0.0, 3.0],
        "TUE": [1.0, 0.0],
        "CALC": ["no", "Sometimes"],
        "MTRANS": ["Public_Transportation", "Public_Transportation"],
    }

    make_prediction(input_data=data_in)
