
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Dict, Any
import pickle


class Model(ABC):
    """Base class for all models."""

    def __init__(self,
                 model_type: Literal["regression", "classification"],
                 hyperparameters: Dict[str, Any] = None) -> None:
        """
        Initializes the model.

        Args:
            model_type (Literal["regression", "classification"]):
            type of the model.
            hyperparameters (Dict[str, Any]): hyperparameters of the model,
            defaulted to an empty dict.
        """
        self.type = model_type
        self.parameters = {}  # we will use it for learned params
        self.hyperparameters = hyperparameters or {}
        self.model = None  # we set this in the init of subclass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains model on input data and target values.

        Args:
            X (np.ndarray): input data - rows = samples & columns = features
            y (np.ndarray): target values.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values from input data.

        Args:
            X(np.ndarray): data given for prediction

        Returns:
            np.ndarray: predicted target values.
        """
        if not self.model:
            raise ValueError("Model is not initialized.")
        return self.model.predict(X)

    def to_artifact(self, name: str) -> Artifact:
        """
        Transforms model data into an artifact, to be saved.

        Args:
            name(str): name for artifact

        Returns:
            Artifact: serialized artifact that contains model data
        """
        # deep copying to prevent changes in original data, as dict data type
        # is mutable
        data = {
            "parameters": deepcopy(self.parameters),
            "hyperparameters": deepcopy(self.hyperparameters),
        }
        serialized_data = pickle.dumps(data)
        return Artifact(name=name, data=serialized_data, type=self.type)

    def __str__(self) -> str:
        """
        Returns the name of the model.
        """
        return self.__class__.__name__
