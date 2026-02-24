from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from smartmemory.models.memory_item import MemoryItem
from smartmemory.stores.base import BaseHandler


class FileHandler(BaseHandler[MemoryItem]):
    """
    Handler for local filesystem resources (file://path/to/file).
    All methods accept and return canonical MemoryItem objects or dicts with at least 'content' and 'item_id'.
    Supports get, add (write), search (list), and delete.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize FileHandler.
        
        Args:
            base_path: Optional base directory to restrict operations to (jail).
                       If None, allows absolute paths (use with caution).
        """
        self.base_path = Path(base_path).resolve() if base_path else None

    def _parse_file_uri(self, uri: str) -> Path:
        """Parse file:// URI to Path object."""
        if uri.startswith('file://'):
            path_str = uri[7:]
            path = Path(path_str)
        else:
            path = Path(uri)
            
        # If path is relative and we have a base_path, resolve against it
        if not path.is_absolute() and self.base_path:
            path = self.base_path / path
            
        path = path.resolve()
        
        if self.base_path:
            # Security check: ensure path is within base_path
            try:
                path.relative_to(self.base_path)
            except ValueError:
                raise ValueError(f"Access denied: Path {path} is outside base directory {self.base_path}")
                
        return path

    def get(self, item: Union[str, MemoryItem], **kwargs) -> Dict[str, Any]:
        """
        Retrieve file content from filesystem.
        
        Args:
            item: MemoryItem or URI string
            **kwargs: Additional args (encoding, etc.)
            
        Returns:
            Dict with content, item_id, metadata
        """
        if isinstance(item, str):
            uri = item
            item_id = item
        elif isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
            item_id = item.item_id
        else:
            uri = item.get('metadata', {}).get('uri') or item.get('item_id') or item.get('uri')
            item_id = item.get('item_id', uri)

        if not uri:
            raise ValueError("No URI provided for file retrieval")

        path = self._parse_file_uri(uri)
        
        try:
            if not path.exists():
                return None
            
            content = path.read_text(encoding=kwargs.get('encoding', 'utf-8'))
            return {
                'content': content,
                'item_id': item_id,
                'metadata': {
                    'uri': f"file://{path}",
                    'path': str(path),
                    'size': path.stat().st_size,
                    'is_file': True
                }
            }
        except Exception as e:
            raise RuntimeError(f"File get failed: {e}")

    def add(self, item: MemoryItem, **kwargs) -> str:
        """
        Write content to filesystem.
        
        Args:
            item: MemoryItem to write
            **kwargs: encoding, overwrite (bool)
            
        Returns:
            URI string
        """
        if not isinstance(item, MemoryItem):
            raise TypeError('FileHandler only accepts MemoryItem objects for add()')
            
        uri = item.metadata.get('uri') or item.item_id
        if not uri:
             # If no URI, maybe construct one from item_id if we have a base_path
             if self.base_path:
                 uri = f"file://{self.base_path}/{item.item_id}"
             else:
                 raise ValueError("No URI provided and no base_path configured")
                 
        path = self._parse_file_uri(uri)
        content = item.content
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        mode = 'w' if kwargs.get('overwrite', True) else 'x'
        
        try:
            path.write_text(content, encoding=kwargs.get('encoding', 'utf-8'))
            return f"file://{path}"
        except FileExistsError:
            raise FileExistsError(f"File {path} already exists and overwrite=False")
        except Exception as e:
            raise RuntimeError(f"File write failed: {e}")

    def update(self, item: MemoryItem, **kwargs) -> str:
        """Alias for add with overwrite=True."""
        return self.add(item, overwrite=True, **kwargs)

    def search(self, query: str, **kwargs) -> List[str]:
        """
        List files matching glob pattern.
        
        Args:
            query: Glob pattern (e.g. "*.txt")
            **kwargs: 'path' (optional search root within base_path)
            
        Returns:
            List of file:// URIs
        """
        search_root = self.base_path
        if kwargs.get('path'):
            search_root = self._parse_file_uri(kwargs.get('path'))
            
        if not search_root:
             # Fallback to current dir if no base_path? Unsafe.
             raise ValueError("Cannot search without a base directory context")
             
        results = []
        for path in search_root.rglob(query):
            if path.is_file():
                results.append(f"file://{path}")
        return results

    def delete(self, item: Union[str, MemoryItem], **kwargs) -> bool:
        """
        Delete file from filesystem.
        """
        if isinstance(item, str):
            uri = item
        elif isinstance(item, MemoryItem):
            uri = item.metadata.get('uri') or item.item_id
        else:
            uri = item.get('metadata', {}).get('uri') or item.get('item_id')

        if not uri:
            return False

        path = self._parse_file_uri(uri)
        try:
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            raise RuntimeError(f"File delete failed: {e}")
