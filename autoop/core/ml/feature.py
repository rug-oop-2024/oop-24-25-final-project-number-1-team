
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    name: str = Field(title="Feature name")
    type: Literal["numerical", "categorical"] = Field(title="Feature type")

    def __str__(self):
        """
        Returns a string representation of the feature, with name and type.

        Returns:
            str: String representation of the feature.
        """
        return f"Feature(name={self.name}, type={self.type})"