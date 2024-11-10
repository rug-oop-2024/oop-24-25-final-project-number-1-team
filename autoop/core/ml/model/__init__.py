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
    "MultipleLinearRegression",
    "DecisionTreeRegression",
    "RidgeRegression"
]

CLASSIFICATION_MODELS = [
    "LogisticRegressionModel",
    "KNeighborsClassifierModel",
    "DecisionTreeClassifierModel"
]


def get_model(model_name: str) -> Model:
    """Factory function to get model by its name"""
    default_line1 = f"Model {model_name} is not supported."
    default_line2 = f"""List of supported models:
    {REGRESSION_MODELS + CLASSIFICATION_MODELS}"""

    match model_name:
        case "MultipleLinearRegression":
            return MultipleLinearRegression()
        case "DecisionTreeRegression":
            return DecisionTreeRegression()
        case "RidgeRegression":
            return RidgeRegression()
        case "LogisticRegressionModel":
            return LogisticRegressionModel()
        case "KNeighborsClassifierModel":
            return KNeighborsClassifierModel()
        case "DecisionTreeClassifierModel":
            return DecisionTreeClassifierModel()
        case _:
            raise ValueError(default_line1 + '\n' + default_line2)
