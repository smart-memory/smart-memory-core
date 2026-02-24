from typing import Any, List, Union

from smartmemory.models.library import Library, Document
from smartmemory.stores.base import BaseHandler
from smartmemory.stores.persistence.base import PersistenceBackend

# Aliases for backward compat — callers that import CorpusStore still work
Corpus = Library


class CorpusStore(BaseHandler[Library]):
    """
    Store for managing Library (formerly Corpus) metadata/Lifecycle.
    Uses a PersistenceBackend to save Library objects.
    """

    def __init__(self, persistence: PersistenceBackend[Library]):
        self.persistence = persistence

    def add(self, item: Library, **kwargs) -> Union[str, Library, None]:
        if not isinstance(item, Library):
            raise TypeError("CorpusStore only accepts Library objects")
        self.persistence.save(item)
        return item.id

    def update(self, item: Library, **kwargs) -> Union[str, Library, None]:
        if not isinstance(item, Library):
            raise TypeError("CorpusStore only accepts Library objects")
        self.persistence.save(item)
        return item.id

    def get(self, item_id: str, **kwargs) -> Union[Library, None]:
        return self.persistence.find_one(Library, id=item_id)

    def delete(self, item_id: str, **kwargs) -> bool:
        return self.persistence.delete_one(Library, id=item_id)

    def search(self, query: Any, **kwargs) -> List[Library]:
        return self.persistence.find_many(Library)


class DocumentStore(BaseHandler[Document]):
    """Store for managing Document metadata."""

    def __init__(self, persistence: PersistenceBackend[Document]):
        self.persistence = persistence

    def add(self, item: Document, **kwargs) -> Union[str, Document, None]:
        if not isinstance(item, Document):
            raise TypeError("DocumentStore only accepts Document objects")
        self.persistence.save(item)
        return item.id

    def update(self, item: Document, **kwargs) -> Union[str, Document, None]:
        if not isinstance(item, Document):
            raise TypeError("DocumentStore only accepts Document objects")
        self.persistence.save(item)
        return item.id

    def get(self, item_id: str, **kwargs) -> Union[Document, None]:
        return self.persistence.find_one(Document, id=item_id)

    def delete(self, item_id: str, **kwargs) -> bool:
        return self.persistence.delete_one(Document, id=item_id)

    def search(self, query: Any, **kwargs) -> List[Document]:
        # Support filtering by library_id (and legacy corpus_id) via kwargs or query
        filters = {}
        if isinstance(query, dict):
            filters.update(query)
        if 'library_id' in kwargs:
            filters['library_id'] = kwargs['library_id']
        if 'corpus_id' in kwargs:
            filters['library_id'] = kwargs['corpus_id']

        return self.persistence.find_many(Document, **filters)
