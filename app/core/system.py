from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """Class used to manage artifacts in our system."""
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """
        Initializes the class with a storage and a database.

        Args:
            database (Database): The database instance that is used 
                                for entries.
            storage (Storage): The storage to use for saving
                                and loading artifacts.
        """
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Register an artifact given its data and metadata in db
        and storage.

        Args:
            artifact (Artifact): The artifact to register.
        """
        # save the artifact in the storage
        self._storage.save(artifact.data, artifact.asset_path)
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all artifacts with filters possiblity by type.

        Args:
            type (str): The type of artifact to filter by; optional.
        Returns:
            List[Artifact]: The list of artifacts.
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact object by its id.

        Args:
            artifact_id (str): The id of the artifact to retrieve.
        Returns:
            Artifact: The artifact object.
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes one artiiifact from storage and db, by its id.

        Args:
            artifact_id (str): The id of the artifact to delete.
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    Singleton class used for managing the artifact registry. It is used
    to access the registry and database, and works as an interface for Artifact
    Registry to ensure there is only one instance of the registry and database
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the class with the local storage for file operations and
        database.

        Args:
            storage (LocalStorage): The storage to use for saving and loading
                                    artifacts.
            database (Database): The database instance that is used for
                                entries.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> 'AutoMLSystem':
        """
        Returns the singleton instance; it creates one if it does not exist.

        Returns:
            AutoMLSystem: The singleton instance.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"),
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
            Gives access to the registry of artifacts.

            Returns:
                ArtifactRegistry: The registry of artifacts.
        """
        return self._registry
