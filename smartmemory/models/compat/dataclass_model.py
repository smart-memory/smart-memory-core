import dataclasses
import json
import re
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Type, TypeVar

T = TypeVar("T")


def _looks_like_datetime(value: str) -> bool:
    """Check if a string looks like an ISO datetime format."""
    # Match ISO 8601 datetime formats like "2024-01-15T10:30:45+00:00" or "2024-01-15 10:30:45+00:00"
    datetime_pattern = r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$'
    return bool(re.match(datetime_pattern, value))


def _parse_datetime(value: str) -> datetime:
    """Parse various datetime string formats back to datetime objects."""
    # Handle common formats from datetime.isoformat()
    try:
        # Try parsing with timezone info
        if value.endswith('Z'):
            value = value[:-1] + '+00:00'

        # Replace space with T for ISO format if needed
        if ' ' in value and 'T' not in value:
            parts = value.split(' ')
            if len(parts) == 2:
                value = f"{parts[0]}T{parts[1]}"

        return datetime.fromisoformat(value)
    except ValueError:
        # Fallback: try without timezone for older formats
        try:
            return datetime.fromisoformat(value.replace('Z', ''))
        except ValueError:
            # Last resort: assume UTC if no timezone
            dt = datetime.fromisoformat(value.split('+')[0].split('Z')[0])
            return dt.replace(tzinfo=timezone.utc)


def _convert_value(v: Any) -> Any:
    # Preserve Python types (datetime, Enum, etc.) to mirror Pydantic to_dict
    # Only recurse into dataclasses, mappings, and sequences
    if is_dataclass(v):
        return _convert_dict(dataclasses.asdict(v))
    if isinstance(v, Mapping):
        return {k: _convert_value(val) for k, val in v.items()}
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return [_convert_value(i) for i in v]
    return v


def _convert_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _convert_value(v) for k, v in d.items()}


def get_field_names(cls: Type[Any]) -> set[str]:
    """Return data field names for dataclasses or Pydantic models."""
    if dataclasses.is_dataclass(cls):
        return {f.name for f in fields(cls)}
    # Pydantic v2
    if hasattr(cls, "model_fields") and isinstance(cls.model_fields, Mapping):
        return set(cls.model_fields.keys())
    # Pydantic v1
    if hasattr(cls, "__fields__") and isinstance(cls.__fields__, Mapping):
        return set(cls.__fields__.keys())
    return set()


class DataclassModelMixin:
    """
    Compatibility mixin that provides common (de)serialization helpers and
    shims for Pydantic-like methods so callers don't break during migration.
    """

    def to_dict(self) -> Dict[str, Any]:
        # Prefer dataclass conversion
        if is_dataclass(self):
            return _convert_dict(dataclasses.asdict(self))
        # Fallbacks for Pydantic models
        if hasattr(self, "to_dict"):
            return self.to_dict()
        if hasattr(self, "dict"):
            return self.dict()
        # Last resort: instance __dict__
        return _convert_dict(dict(self.__dict__))

    def to_json(self) -> str:
        def _default(o: Any):
            if isinstance(o, (datetime, date, time)):
                return o.isoformat()
            if isinstance(o, Enum):
                return o.value
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, set):
                return list(o)
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        return json.dumps(self.to_dict(), ensure_ascii=False, default=_default)

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        names = get_field_names(cls)
        data = {k: v for k, v in d.items() if k in names} if names else d

        # Parse datetime strings back to datetime objects
        parsed_data = {}
        for key, value in data.items():
            if isinstance(value, str) and _looks_like_datetime(value):
                try:
                    parsed_data[key] = _parse_datetime(value)
                except (ValueError, TypeError):
                    parsed_data[key] = value  # Keep as string if parsing fails
            else:
                parsed_data[key] = value

        return cls(**parsed_data)  # type: ignore[misc]

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Generate a minimal JSON Schema for a dataclass or Pydantic model.
        Prefers dataclass introspection. Nested dataclasses supported.
        """
        from typing import get_origin, get_args, Union, List, Dict as TDict
        from dataclasses import MISSING

        def _pytype_to_schema(tp: Any) -> Dict[str, Any]:
            origin = get_origin(tp)
            args = get_args(tp)

            # Optional[T] / Union[T, None]
            if origin is Union and type(None) in args:
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                base = _pytype_to_schema(non_none[0]) if non_none else {"type": "object"}
                base["nullable"] = True
                return base

            # List[T]
            if origin in (list, List):
                item_tp = args[0] if args else Any
                return {"type": "array", "items": _pytype_to_schema(item_tp)}

            # Dict[K,V]
            if origin in (dict, TDict):
                return {"type": "object", "additionalProperties": True}

            # datetime/date/time
            if tp is datetime:
                return {"type": "string", "format": "date-time"}
            if tp is date:
                return {"type": "string", "format": "date"}
            if tp is time:
                return {"type": "string", "format": "time"}

            # Nested dataclass
            if is_dataclass(tp):
                return _dataclass_schema(tp)

            # Primitives
            if tp in (str,):
                return {"type": "string"}
            if tp in (int,):
                return {"type": "integer"}
            if tp in (float,):
                return {"type": "number"}
            if tp in (bool,):
                return {"type": "boolean"}

            # Fallback
            return {"type": "object"}

        def _dataclass_schema(dc_cls: Type[Any]) -> Dict[str, Any]:
            props: Dict[str, Any] = {}
            required: list[str] = []
            for f in fields(dc_cls):
                props[f.name] = _pytype_to_schema(f.type)
                has_default = (f.default is not MISSING) or (getattr(f, 'default_factory', MISSING) is not MISSING)
                if not has_default:
                    required.append(f.name)
            schema: Dict[str, Any] = {"type": "object", "title": dc_cls.__name__, "properties": props}
            if required:
                schema["required"] = required
            return schema

        # Dataclass preferred
        if is_dataclass(cls):
            return _dataclass_schema(cls)

        # Pydantic fallback (v2 preferred)
        if hasattr(cls, "model_json_schema") and callable(cls.model_json_schema):
            try:
                return cls.model_json_schema()
            except Exception:
                pass
        if hasattr(cls, "schema") and callable(cls.schema):
            try:
                return cls.schema()
            except Exception:
                pass

        # Last resort
        return {"type": "object", "title": getattr(cls, "__name__", "Model")}

    def to_pydantic(self, pydantic_class=None):
        """
        Automatic conversion to Pydantic models - behind the scenes boundary conversion.
        
        Usage: 
            concept_model = concept_ir.to_pydantic(ConceptModel)  # Explicit type
            concept_model = concept_ir.to_pydantic()             # Auto-inferred type
        """
        if pydantic_class is None:
            # Auto-infer Pydantic class from AgenticBaseModel class name
            pydantic_class = self._infer_pydantic_class()

        return pydantic_class(**self.to_dict())

    def _infer_pydantic_class(self):
        """
        Create a Pydantic class dynamically from the MemoryBaseModel dataclass definition.
        Recursively converts nested MemoryBaseModel fields to Pydantic as well.
        """
        from pydantic import BaseModel, Field
        from dataclasses import fields, is_dataclass

        if not is_dataclass(self.__class__):
            raise ValueError(f"{self.__class__.__name__} is not a dataclass")

        # Extract field information from dataclass
        field_definitions = {}
        annotations = {}

        for field in fields(self.__class__):
            field_type = field.type
            default_value = field.default if field.default != field.default_factory else field.default_factory()

            # Extract description from metadata if available
            description = field.metadata.get("description", "")

            # Handle nested MemoryBaseModel types recursively
            converted_type = self._convert_nested_types(field_type)

            # Set up annotations
            annotations[field.name] = converted_type

            # Create Pydantic Field
            if description:
                field_definitions[field.name] = Field(default=default_value, description=description)
            else:
                field_definitions[field.name] = default_value

        # Create dynamic Pydantic models class
        return type(
            self.__class__.__name__,
            (BaseModel,),
            {
                "__annotations__": annotations,
                **field_definitions
            }
        )

    def _convert_nested_types(self, field_type):
        """
        Recursively convert nested MemoryBaseModel types to their Pydantic equivalents.
        """
        from dataclasses import is_dataclass
        from typing import get_origin, get_args
        import typing

        # Handle List[SomeDataclass] -> List[SomePydanticModel]
        origin = get_origin(field_type)
        if origin is list or origin is typing.List:
            args = get_args(field_type)
            if args and is_dataclass(args[0]) and hasattr(args[0], '_infer_pydantic_class'):
                # Create instance to get the Pydantic class
                temp_instance = args[0]()
                nested_pydantic_class = temp_instance._infer_pydantic_class()
                return typing.List[nested_pydantic_class]

        # Handle direct dataclass fields
        elif is_dataclass(field_type) and hasattr(field_type, '_infer_pydantic_class'):
            # Create instance to get the Pydantic class
            temp_instance = field_type()
            return temp_instance._infer_pydantic_class()

        # Return original type if no conversion needed
        return field_type

    @classmethod
    def from_pydantic(cls, pydantic_model):
        """
        Automatic conversion from Pydantic models - behind the scenes boundary conversion.
        
        Usage: concept_ir = ConceptIR.from_pydantic(concept_model)
        """
        return cls.from_dict(pydantic_model.dict())
