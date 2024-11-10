from pydantic import BaseModel, Field
from typing import List, Dict
import base64


class Artifact(BaseModel):
    """A class used to store info about an artifact, for example its name,
    version, asset path, metadata, tags, type. It also stores the binary data
    of the artifact. It also has a property for generating an uniquee id, as
    per requirement.
    """
    name: str = Field(title="Artifact name")
    version: str = Field(title="Artifact version")
    asset_path: str = Field(title="Artifact asset path")
    data: bytes = Field(title="Data of artifact5 in binary format")
    metadata: Dict[str, str] = Field(default_factory=dict,
                                     title="Artifact metadata")
    tags: List[str] = Field(default_factory=list, title="Artifact tags")
    type: str = Field(title="Artifact type")

    @property
    def id(self) -> str:
        """
        Generates an unique ID of the artifact, by encoding the asset path
        and appending the version.

        Returns:
            str: Unique ID of the artifact.
        """
        base_path = base64.b64encode(self.asset_path.encode()).decode()
        encoded_path = base_path.rstrip("=").replace('+', '-').replace('/', '_')
        encoded_version = self.version.replace('.', '_').replace(":", "_").replace(',', '_').replace('=', '_')
        return f"{encoded_path}_{encoded_version}"

    def read(self) -> bytes:
        """
        Returns the binary data of the artifact. As bytes data type
        is immutable, we can directly return it without risk of
        data leakage.

        Returns:
            bytes: Binary data of the artifact.
        """
        return self.data

    def save(self, data: bytes) -> bytes:
        """
        Saves binary data to the artifact, then returns it.

        Args:
            data (bytes): Binary data to save.
        Returns:
            bytes: Saved binary data.
        """
        self.data = data
        return self.data
