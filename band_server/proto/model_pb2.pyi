from proto import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelDescriptor(_message.Message):
    __slots__ = ["id", "input_tensor_indices", "name", "num_ops", "num_tensors", "op_input_tensors", "op_output_tensors", "output_tensor_indices", "tensor_types"]
    ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_INDICES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_OPS_FIELD_NUMBER: _ClassVar[int]
    NUM_TENSORS_FIELD_NUMBER: _ClassVar[int]
    OP_INPUT_TENSORS_FIELD_NUMBER: _ClassVar[int]
    OP_OUTPUT_TENSORS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_INDICES_FIELD_NUMBER: _ClassVar[int]
    TENSOR_TYPES_FIELD_NUMBER: _ClassVar[int]
    id: str
    input_tensor_indices: _containers.RepeatedScalarFieldContainer[int]
    name: str
    num_ops: int
    num_tensors: int
    op_input_tensors: _containers.RepeatedCompositeFieldContainer[_common_pb2.OpSet]
    op_output_tensors: _containers.RepeatedCompositeFieldContainer[_common_pb2.OpSet]
    output_tensor_indices: _containers.RepeatedScalarFieldContainer[int]
    tensor_types: _containers.RepeatedScalarFieldContainer[_common_pb2.DataType]
    def __init__(self, name: _Optional[str] = ..., id: _Optional[str] = ..., num_ops: _Optional[int] = ..., num_tensors: _Optional[int] = ..., tensor_types: _Optional[_Iterable[_Union[_common_pb2.DataType, str]]] = ..., input_tensor_indices: _Optional[_Iterable[int]] = ..., output_tensor_indices: _Optional[_Iterable[int]] = ..., op_input_tensors: _Optional[_Iterable[_Union[_common_pb2.OpSet, _Mapping]]] = ..., op_output_tensors: _Optional[_Iterable[_Union[_common_pb2.OpSet, _Mapping]]] = ...) -> None: ...
