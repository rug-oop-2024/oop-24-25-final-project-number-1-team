from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression
import numpy as np

class MultipleLinearRegression(Model):
    """Multiple Linear Regression model."""

    def __init__(self, **hyperparameters):
        super().__init__(model_type="regression", hyperparameters=hyperparameters)
        self.model = LinearRegression(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.parameters = self.model.coef_