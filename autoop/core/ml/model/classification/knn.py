from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNeighborsClassifierModel(Model):
    """K-Nearest Neighbors Classifier model"""

    def __init__(self, **hyperparameters):
        super().__init__(model_type="classification", hyperparameters=hyperparameters)
        self.model = KNeighborsClassifier(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.parameters = self.model.get_params()