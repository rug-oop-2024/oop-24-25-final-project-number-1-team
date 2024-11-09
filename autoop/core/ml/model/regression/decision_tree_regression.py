from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class DecisionTreeRegression(Model):
    """Decision Tree Regression model."""

    def __init__(self, **hyperparameters):
        super().__init__(model_type="regression", hyperparameters=hyperparameters)
        self.model = DecisionTreeRegressor(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.parameters = self.model.get_params()