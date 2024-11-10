from autoop.core.ml.model import Model
from sklearn.linear_model import Ridge
import numpy as np


class RidgeRegression(Model):
    """Ridge Regression model."""

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the model and sets hyperparameters.
        """
        super().__init__(model_type="regression",
                         hyperparameters=hyperparameters)
        self.model = Ridge(**hyperparameters)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains model on input data and target values.

        Args:
            X (np.ndarray): input data - rows = samples & columns = features
            y (np.ndarray): target values.
        """
        self.model.fit(X, y)
        self.parameters = self.model.coef_
