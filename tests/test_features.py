"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
import math
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from obesity_model.config.core import config
from obesity_model.processing.features import OutlierHandler, CategoricalOneHotEncoder


def test_outlier_handler(sample_input_data):
    # Given
    transformer = OutlierHandler(
        iqr_multiplier=config.model_config.iqr_multiplier,
    )
    assert sample_input_data["Age"].max() > 35.375

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert subject["Age"].max().round(3) == 35.375


def test_categorical_onehot_encoder(sample_input_data):
    # Given
    transformer = CategoricalOneHotEncoder(
        categorical_feature= config.model_config.gender_column,  # cabin
    )
    assert len(sample_input_data.columns) == 17

    # When
    subject = transformer.fit(sample_input_data).transform(sample_input_data)

    # Then
    assert len(subject.columns) == 18
