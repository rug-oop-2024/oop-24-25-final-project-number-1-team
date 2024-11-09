from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionModel(Model):
    """Logistic Regression model"""

    def __init__(self, **hyperparameters):
        super().__init__(model_type="classification", hyperparameters=hyperparameters)
        self.model = LogisticRegression(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.parameters = self.model.coef_