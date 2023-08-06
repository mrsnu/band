import tensorrt as trt
from proto import DataType, Shape

DTYPE_MAP = {
    trt.DataType.FLOAT: DataType.FLOAT32,
    trt.DataType.HALF: DataType.FLOAT16,
    trt.DataType.INT8: DataType.INT8,
    trt.DataType.INT32: DataType.INT32,
    trt.DataType.BOOL: DataType.BOOL,
}

SIZE_MAP = {
    trt.DataType.FLOAT: 4,
    trt.DataType.HALF: 2,
    trt.DataType.INT8: 1,
    trt.DataType.INT32: 4,
    trt.DataType.BOOL: 1,
}

def get_bytes_size(dtype):
    return SIZE_MAP[dtype]

def convert_dtype(dtype):
    return DTYPE_MAP[dtype]

def convert_shape(shape):
    return Shape(dims=list(shape))