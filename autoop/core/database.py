
import json
from typing import Dict, Tuple, List, Union
import os

from autoop.core.storage import Storage


class Database():
    """A database class in the form of key-value store"""

    def __init__(self, storage: Storage) -> None:
        """
        Initialize the database with a given storage interface

        Args:
            storage (Storage): The storage interface for saving loading data
        """
        self._storage = storage
        self._data: Dict[str, Dict[str, dict]] = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """Set a key in the database
        Args:
            collection (str): The collection to store the data in
            id (str): The id of the data
            entry (dict): The data to store
        Returns:
            dict: The data that was stored
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """Get a key from the database
        Args:
            collection (str): The collection to get the data from
            id (str): The id of the data
        Returns:
            Union[dict, None]: The data that was stored, or None
                               if it doesn't exist
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """Delete a key from the database
        Args:
            collection (str): The collection to delete the data from
            id (str): The id of the data
        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """Lists all data in a collection
        Args:
            collection (str): The collection to list the data from
        Returns:
            List[Tuple[str, dict]]: A list of tuples containing the id and data
                                    for each item in the collection
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Refresh the database by loading the data from storage"""
        self._load()

    def _persist(self) -> None:
        """Persist the data to storage"""
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(json.dumps(item).encode(),
                                   os.path.join(collection, id))

        # for things that were deleted, we need to remove them from the storage
        keys = self._storage.list("")
        for key in keys:
            norm_key = os.path.normpath(key)  # for os compatibility
            parts = norm_key.split(os.sep)
            if len(parts) < 2:
                continue
            collection, id = parts[-2:]
            if not self._data.get(collection, {}).get(id):
                self._storage.delete(os.path.join(collection, id))

    def _load(self) -> None:
        """Load the data from storage"""
        self._data = {}
        for key in self._storage.list(""):
            norm_key = os.path.normpath(key)  # for os compatibility
            parts = norm_key.split(os.sep)
            if len(parts) < 2:
                continue
            collection, id = parts[-2:]
            data = self._storage.load(os.path.join(collection, id))
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
