import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from obesity_model.config.core import config
from obesity_model.processing.features import OutlierHandler
from obesity_model.processing.features import Mapper
from obesity_model.processing.features import CategoricalOneHotEncoder

columns_to_drop = ["dteday", "weekday"]
obesity_pipe = Pipeline(
    [
        # Outlier Handling
        (
            "outlier_handler",
            OutlierHandler(iqr_multiplier=config.model_config.iqr_multiplier),
        ),
        # One-Hot Encoding
        (
            "gender_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.gender_column
            ),
        ),
        (
            "family_history_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.family_history_column
            ),
        ),
        (
            "favc_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.favc_column
            ),
        ),
        (
            "caec_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.caec_column
            ),
        ),
        (
            "smoke_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.smoke_column
            ),
        ),
        (
            "scc_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.scc_column
            ),
        ),
        (
            "calc_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.calc_column
            ),
        ),
        (
            "mtrans_encoder",
            CategoricalOneHotEncoder(
                categorical_feature=config.model_config.mtrans_column
            ),
        ),
        # Feature Scaling (optional)
        ("scaler", MinMaxScaler()),
        # Regression Model
        (
            "model_rf",
            RandomForestClassifier(
                n_estimators=config.model_config.n_estimators,
                max_depth=config.model_config.max_depth,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
