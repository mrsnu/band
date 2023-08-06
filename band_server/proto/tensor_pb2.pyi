from proto import common_pb2 as _common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AffineQuantizationParam(_message.Message):
    __slots__ = ["quantized_dimension", "scale", "zero_point"]
    QUANTIZED_DIMENSION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    ZERO_POINT_FIELD_NUMBER: _ClassVar[int]
    quantized_dimension: int
    scale: _containers.RepeatedScalarFieldContainer[float]
    zero_point: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, scale: _Optional[_Iterable[float]] = ..., zero_point: _Optional[_Iterable[int]] = ..., quantized_dimension: _Optional[int] = ...) -> None: ...

class Quantization(_message.Message):
    __slots__ = ["affine_param", "type"]
    AFFINE_PARAM_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    affine_param: AffineQuantizationParam
    type: _common_pb2.QuantizationType
    def __init__(self, type: _Optional[_Union[_common_pb2.QuantizationType, str]] = ..., affine_param: _Optional[_Union[AffineQuantizationParam, _Mapping]] = ...) -> None: ...

class Tensor(_message.Message):
    __slots__ = ["data", "dtype", "name", "quantization", "shape"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    dtype: _common_pb2.DataType
    name: str
    quantization: Quantization
    shape: _common_pb2.Shape
    def __init__(self, name: _Optional[str] = ..., shape: _Optional[_Union[_common_pb2.Shape, _Mapping]] = ..., dtype: _Optional[_Union[_common_pb2.DataType, str]] = ..., data: _Optional[bytes] = ..., quantization: _Optional[_Union[Quantization, _Mapping]] = ...) -> None: ...
