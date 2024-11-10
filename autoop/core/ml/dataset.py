from autoop.core.ml.artifact import Artifact
# from abc import ABC, abstractmethod
import pandas as pd
import io


class Dataset(Artifact):
    """A dataset artifact, that allows initialization of an artifact directly
    from a pandas df. It has dataset specific methods, such as read and save.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the artifact, with type set to database, passing all args
        and sets data to an empty byte string.
        """
        super().__init__(type="dataset", *args, **kwargs)
        self._data = b""

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       name: str,
                       asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
        """
        Initializes a dataset artifact from a pandas dataframe.

        Args:
            data (pd.DataFrame): Dataframe to initialize the artifact with.
            name (str): Name of the artifact.
            asset_path (str): Asset path of the artifact.
            version (str, optional): Version of the artifact,
                                    defaulted to "1.0.0".

        Returns:
            Dataset: Initialized dataset artifact.
        """
        instance = Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
        instance._data = data.to_csv(index=False).encode()
        return instance

    def read(self) -> pd.DataFrame:
        """
        Reads the artifact, decodes it then returns it.

        Returns:
            pd.DataFrame: Dataframe read from the artifact.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves a df to the dataset, then returns it.

        Args:
            data (pd.DataFrame): Dataframe to save.

        Returns:
            bytes: Saved binary data.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
