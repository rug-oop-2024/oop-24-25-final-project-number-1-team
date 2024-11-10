"""
This module contains a factory function and constants for showing and
initializing supported models, both regression and classification.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    DecisionTreeRegression,
    RidgeRegression
)
from autoop.core.ml.model.classification import (
    LogisticRegressionModel,
    KNeighborsClassifierModel,
    DecisionTreeClassifierModel
)

REGRESSION_MODELS = [
    "Multiple Linear Regression",
    "Decision Tree Regression",
    "Ridge Regression"
]

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree"
]


def get_model(model_name: str) -> Model:
    """Factory function to get model by its name"""
    default_line1 = f"Model {model_name} is not supported."
    default_line2 = f"""List of supported models:
    {REGRESSION_MODELS + CLASSIFICATION_MODELS}"""

    match model_name:
        case "Multiple Linear Regression":
            return MultipleLinearRegression()
        case "Decision Tree Regression":
            return DecisionTreeRegression()
        case "Ridge Regression":
            return RidgeRegression()
        case "Logistic Regression":
            return LogisticRegressionModel()
        case "K-Nearest Neighbors":
            return KNeighborsClassifierModel()
        case "Decision Tree":
            return DecisionTreeClassifierModel()
        case _:
            raise ValueError(default_line1 + '\n' + default_line2)
