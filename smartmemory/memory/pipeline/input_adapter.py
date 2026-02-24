"""
InputAdapter component for componentized memory ingestion pipeline.
Handles conversion of raw input to MemoryItem with metadata normalization.
"""
from datetime import datetime
from typing import Dict, Any, Optional

from smartmemory.memory.pipeline.components import PipelineComponent, ComponentResult
from smartmemory.memory.pipeline.config import InputAdapterConfig
from smartmemory.models.memory_item import MemoryItem


class InputAdapter(PipelineComponent):
    """
    Component responsible for converting raw input to standardized MemoryItem format.
    Handles various input types and normalizes metadata for downstream processing.
    """

    def __init__(self):
        self.adapter_registry = {}
        self._register_default_adapters()

    def _register_default_adapters(self):
        """Register default input adapters for common formats"""
        self.adapter_registry['text'] = self._text_adapter
        self.adapter_registry['dict'] = self._dict_adapter
        self.adapter_registry['json'] = self._json_adapter
        self.adapter_registry['default'] = self._default_adapter

    def register_adapter(self, name: str, adapter_fn):
        """Register a custom input adapter"""
        self.adapter_registry[name] = adapter_fn

    def _text_adapter(self, item: str) -> MemoryItem:
        """Convert plain text to MemoryItem"""
        return MemoryItem(content=item, memory_type='semantic')

    def _dict_adapter(self, item: Dict[str, Any]) -> MemoryItem:
        """Convert dictionary to MemoryItem"""
        return MemoryItem(**item)

    def _json_adapter(self, item: str) -> MemoryItem:
        """Convert JSON string to MemoryItem"""
        import json
        try:
            data = json.loads(item)
            return MemoryItem(**data) if isinstance(data, dict) else MemoryItem(content=str(data))
        except json.JSONDecodeError:
            return MemoryItem(content=item, memory_type='semantic')

    def _default_adapter(self, item: Any) -> MemoryItem:
        """Default adapter for any input type"""
        if isinstance(item, MemoryItem):
            return item
        elif isinstance(item, dict):
            return MemoryItem(**item)
        else:
            return MemoryItem(content=str(item), memory_type='semantic')

    def _normalize_metadata(self, item: MemoryItem) -> MemoryItem:
        """Normalize metadata fields with timestamps and status"""
        now = datetime.now()

        if not hasattr(item, 'metadata') or item.metadata is None:
            item.metadata = {}

        # Set creation time if not present
        item.metadata.setdefault('created_at', now)

        # Always update modification time
        item.metadata['updated_at'] = now

        # Set default status
        item.metadata.setdefault('status', 'created')

        # Set default memory type
        if not item.memory_type:
            item.memory_type = 'semantic'

        return item

    def validate_config(self, config: InputAdapterConfig) -> bool:
        """Validate InputAdapter configuration"""
        if not isinstance(config, dict):
            try:
                config = config.to_dict()
            except Exception:
                return False

        adapter_name = config.get('adapter_name', 'default')
        return adapter_name in self.adapter_registry

    def run(self, input_state: Optional[ComponentResult], config: InputAdapterConfig) -> ComponentResult:
        """
        Execute InputAdapter with given configuration.
        
        Args:
            input_state: None (InputAdapter is the first stage)
            config: Configuration dict with 'input_data' and optional 'adapter_name'
        
        Returns:
            ComponentResult with MemoryItem and metadata
        """
        self.validate_config(config)

        try:
            # Extract input data and adapter name from config
            input_data = config.content
            if input_data is None:
                return ComponentResult(
                    success=False,
                    data={'error': 'No content or input_data provided in config'},
                    metadata={'stage': 'input_adapter'}
                )

            # Get appropriate adapter
            adapter_name = config.adapter_name if config.adapter_name else 'default'
            if adapter_name not in self.adapter_registry:
                adapter_name = 'default'

            adapter_fn = self.adapter_registry[adapter_name]

            # Convert input to MemoryItem
            memory_item = adapter_fn(input_data)
            
            # Merge config metadata into item metadata (preserves scope from ScopedPipeline)
            if config.metadata:
                if not memory_item.metadata:
                    memory_item.metadata = {}
                memory_item.metadata.update(config.metadata)
            
            # Apply memory_type from config if specified
            if config.memory_type:
                memory_item.memory_type = config.memory_type

            # Normalize metadata
            memory_item = self._normalize_metadata(memory_item)

            return ComponentResult(
                success=True,
                data={
                    'memory_item': memory_item,
                    'input_metadata': {
                        'adapter_used': adapter_name,
                        'input_type': type(input_data).__name__,
                        'content_length': len(str(input_data)),
                        'memory_type': memory_item.memory_type
                    }
                },
                metadata={
                    'stage': 'input_adapter',
                    'adapter_name': adapter_name,
                    'normalized': True
                }
            )

        except Exception as e:
            return ComponentResult(
                success=False,
                data={'error': str(e)},
                metadata={
                    'stage': 'input_adapter',
                    'error_type': type(e).__name__,
                    'adapter_name': config.adapter_name
                }
            )
