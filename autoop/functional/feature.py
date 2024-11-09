
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    This function detects feature types in a given dataset, and returns a list
    of features with their types.

    Args:
        dataset (Dataset): Dataset object to analyze.
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    df = dataset.read()

    # Loop over columns in the dataset
    for column in df.columns:
        # From assumption, they are either numerical or categorical
        if pd.api.types.is_numeric_dtype(df[column]):
            type = "numerical"
        else:
            type = "categorical"

        # Append feature to the list.
        features.append(Feature(name=column, type=type))

    # As seen in the test, we return the list of features with name and type.
    return features