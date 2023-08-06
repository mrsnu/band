from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

AFFINE_QUANTIZATION: QuantizationType
BOOL: DataType
COMPLEX64: DataType
DESCRIPTOR: _descriptor.FileDescriptor
FLOAT16: DataType
FLOAT32: DataType
FLOAT64: DataType
INT16: DataType
INT32: DataType
INT64: DataType
INT8: DataType
INVALID_ARGUMENT: StatusCode
NO_QUANTIZATION: QuantizationType
NO_TYPE: DataType
OK: StatusCode
STRING: DataType
UINT8: DataType
UNKNOWN_ERROR: StatusCode

class OpSet(_message.Message):
    __slots__ = ["op"]
    OP_FIELD_NUMBER: _ClassVar[int]
    op: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, op: _Optional[_Iterable[int]] = ...) -> None: ...

class Shape(_message.Message):
    __slots__ = ["dims"]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...

class Status(_message.Message):
    __slots__ = ["code", "error_message"]
    CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: StatusCode
    error_message: str
    def __init__(self, code: _Optional[_Union[StatusCode, str]] = ..., error_message: _Optional[str] = ...) -> None: ...

class Void(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class QuantizationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class StatusCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
