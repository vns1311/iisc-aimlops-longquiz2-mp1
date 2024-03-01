from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from obesity_model.config.core import config


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variables: str, mappings: dict):
        # YOUR CODE HERE
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # YOUR CODE HERE
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # YOUR CODE HERE
        X_copy = X.copy()
        X_copy[self.variables] = X_copy[self.variables].map(self.mappings).astype(int)
        return X_copy


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, numerical_columns=None, iqr_multiplier=1.5):
        # YOUR CODE HERE
        self.numerical_columns = (
            numerical_columns
            if numerical_columns
            else config.model_config.numerical_columns
        )
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        # YOUR CODE HERE
        self.iqr = X[self.numerical_columns].quantile(0.75) - X[
            self.numerical_columns
        ].quantile(0.25)
        return self

    def transform(self, X):
        # YOUR CODE HERE
        X_copy = X.copy()

        for column in self.numerical_columns:
            # Define upper and lower bounds based on IQR
            upper_bound = (
                X_copy[column].quantile(0.75) + self.iqr_multiplier * self.iqr[column]
            )
            lower_bound = (
                X_copy[column].quantile(0.25) - self.iqr_multiplier * self.iqr[column]
            )

            # Cap values to upper and lower bounds
            X_copy[column] = X_copy[column].clip(lower=lower_bound, upper=upper_bound)
        return X_copy


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encode categorical column"""

    def __init__(self, categorical_feature):
        # YOUR CODE HERE
        self.feature = categorical_feature
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        self.one_hot_encoder.fit(X[[self.feature]])
        # Get encoded feature names
        self.encoded_features_names = self.one_hot_encoder.get_feature_names_out(
            [self.feature]
        )

        return self

    def transform(self, X):
        X = X.copy()

        encoded_features = self.one_hot_encoder.transform(X[[self.feature]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_features

        # drop 'weekday' column after encoding
        X.drop(self.feature, axis=1, inplace=True)

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.drop(columns=self.columns_to_drop, inplace=True)

        return X_copy
