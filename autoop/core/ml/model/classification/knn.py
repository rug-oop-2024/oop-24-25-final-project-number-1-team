from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNeighborsClassifierModel(Model):
    """K-Nearest Neighbors Classifier model"""

    def __init__(self, **hyperparameters):
        """
        Initializes the classification model and passes hyperparameters
        """
        super().__init__(model_type="classification",
                         hyperparameters=hyperparameters)
        self.model = KNeighborsClassifier(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains model on input data and target values.

        Args:
            X (np.ndarray): input data - rows = samples & columns = features
            y (np.ndarray): target values.
        """
        self.model.fit(X, y)
        self.parameters = self.model.get_params()
