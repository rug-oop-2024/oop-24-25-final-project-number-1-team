from abc import ABC, abstractmethod
import numpy as np

REGRESSION_METRICS = [
    "Mean Squared Error",
    "Mean Absolute Error",
    "R2 Score",
]

CLASSIFICATION_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
]

METRICS = REGRESSION_METRICS + CLASSIFICATION_METRICS

def get_metric(name: str) -> 'Metric':
    """
        Factory function to get a metric by name.

        Args:
            name (str): name of the metric to get.
    """
    match name:
        case "Mean Squared Error":
            return MeanSquaredError()
        case "Mean Absolute Error":
            return MeanAbsoluteError()
        case "R2 Score":
            return RSquared()
        case "Accuracy":
            return Accuracy()
        case "Precision":
            return Precision()
        case "Recall":
            return Recall()
        case _:
            raise ValueError(f"Metric {name} is not available. "
                             f"List of available metrics: {METRICS}.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the metric from the predictions and ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        pass

    def evaluate(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Alias for __call__ method, as it is needed in pipeline's _evaluate

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.
        Returns:
            float: value of the metric.
        """
        return self.__call__(predictions, ground_truth)

    def __str__(self) -> str:
        """
        Used in pipeline to print the metric name

        Returns:
            str: name of the metric.
        """
        return self.__class__.__name__


# Regression Metrics


class MeanSquaredError(Metric):
    """Mean squared Error (MSE) metric."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the MSE metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        return np.mean((predictions - ground_truth) ** 2)


class MeanAbsoluteError(Metric):
    """Mean Absolute Error (MAE) metric."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the MAE metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        return np.mean(np.abs(predictions - ground_truth))


class RSquared(Metric):
    """R-squared metric."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the rsquared metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        mean = np.mean(ground_truth)
        total = np.sum((ground_truth - mean) ** 2)
        res = np.sum((ground_truth - predictions) ** 2)
        return 1 - (res / total)


# Classification Metrics


class Accuracy(Metric):
    """Accuracy metric for classifications."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the accuracy metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        return np.mean(predictions == ground_truth)


class Precision(Metric):
    """Precision metric for classifications."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the precision metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        classes = np.unique(ground_truth)
        precisions = []
        for c in classes:
            tp = np.sum((predictions == c) & (ground_truth == c))
            fp = np.sum((predictions == c) & (ground_truth != c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return np.mean(precisions)


class Recall(Metric):
    """Recall metric for classifications."""

    def __call__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates the recall metric from the predictions and
        ground truth values.

        Args:
            predictions (np.ndarray): predictions of the model.
            ground_truth (np.ndarray): ground truth values.

        Returns:
            float: value of the metric.
        """
        classes = np.unique(ground_truth)
        recalls = []
        for c in classes:
            tp = np.sum((predictions == c) & (ground_truth == c))
            fn = np.sum((predictions != c) & (ground_truth == c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return np.mean(recalls)
