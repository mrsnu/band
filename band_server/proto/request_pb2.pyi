from proto import tensor_pb2 as _tensor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

CUSTOM: BackendType
DESCRIPTOR: _descriptor.FileDescriptor
ONNXRUNTIME: BackendType
PYTORCH: BackendType
TENSORFLOW: BackendType
TENSORRT: BackendType

class Request(_message.Message):
    __slots__ = ["option", "tensors"]
    OPTION_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    option: RequestOption
    tensors: _containers.RepeatedCompositeFieldContainer[_tensor_pb2.Tensor]
    def __init__(self, option: _Optional[_Union[RequestOption, _Mapping]] = ..., tensors: _Optional[_Iterable[_Union[_tensor_pb2.Tensor, _Mapping]]] = ...) -> None: ...

class RequestOption(_message.Message):
    __slots__ = ["backend_type"]
    BACKEND_TYPE_FIELD_NUMBER: _ClassVar[int]
    backend_type: BackendType
    def __init__(self, backend_type: _Optional[_Union[BackendType, str]] = ...) -> None: ...

class BackendType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
