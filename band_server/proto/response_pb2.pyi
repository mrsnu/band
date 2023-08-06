from proto import common_pb2 as _common_pb2
from proto import tensor_pb2 as _tensor_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Event(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["status", "tensors"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    status: _common_pb2.Status
    tensors: _containers.RepeatedCompositeFieldContainer[_tensor_pb2.Tensor]
    def __init__(self, status: _Optional[_Union[_common_pb2.Status, _Mapping]] = ..., tensors: _Optional[_Iterable[_Union[_tensor_pb2.Tensor, _Mapping]]] = ...) -> None: ...
