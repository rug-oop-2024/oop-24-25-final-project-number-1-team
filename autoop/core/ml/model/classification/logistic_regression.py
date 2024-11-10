from autoop.core.ml.model import Model
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegressionModel(Model):
    """Logistic Regression model"""

    def __init__(self, **hyperparameters) -> None:
        """
        Initializes the classification model and passes hyperparameters
        """
        super().__init__(model_type="classification",
                         hyperparameters=hyperparameters)
        self.model = LogisticRegression(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains model on input data and target values.

        Args:
            X (np.ndarray): input data - rows = samples & columns = features
            y (np.ndarray): target values.
        """
        self.model.fit(X, y)
        self.parameters = self.model.coef_
