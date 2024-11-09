from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DecisionTreeClassifierModel(Model):
    """Decision Tree Classifier model"""

    def __init__(self, **hyperparameters):
        super().__init__(model_type="classification", hyperparameters=hyperparameters)
        self.model = DecisionTreeClassifier(**hyperparameters)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.parameters = self.model.feature_importances_