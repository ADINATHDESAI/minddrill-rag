"""ORM models.

Importing this package registers every table on `Base.metadata`, so a foreign
key resolves no matter which model a caller imports first. Importing any
submodule runs this, so entry points that only need one model (e.g. the worker
importing `IngestionJob`) still get the whole schema registered.
"""

from minddrill.models.chunk import Chunk
from minddrill.models.document import Document
from minddrill.models.ingestion_job import IngestionJob
from minddrill.models.user import User

__all__ = ["Chunk", "Document", "IngestionJob", "User"]
