from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
        Raised when a path is not found on storage operations.
    """
    def __init__(self, path: str) -> None:
        """
        Initializing NotFoundError with a message.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Creates a LocalStorage instance, that has a base path for handling
        file operations.

        Args:
            base_path (str): Base path for the storage, defaults to './assets'.
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save the data at a given location at key.

        Args:
            data (bytes): Data to save.
            key (str): Location to save the data.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load the file at the given location in the storage.

        Args:
            key (str): File to load.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete the file at the given location in the storage.

        Args:
            key (str): File to delete, defaults to '/'.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all file paths under the given prefix in the storage.

        Args:
            prefix (str): Dir path prefix to list files under, defaults to '/'.

        Returns:
            List[str]: List of file paths under the given prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in
                keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None | NotFoundError:
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
