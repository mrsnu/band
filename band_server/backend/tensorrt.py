import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

from proto import ModelDescriptor
from backend.util import (
    convert_dtype,
    convert_shape,
    get_bytes_size
)

# num_bindings – int The number of binding indices.

# num_io_tensors – int The number of IO tensors.

# max_batch_size – int [DEPRECATED] The maximum batch size which can be used for inference for an engine built from an INetworkDefinition with implicit batch dimension. For an engine built from an INetworkDefinition with explicit batch dimension, this will always be 1 .

# has_implicit_batch_dimension – bool Whether the engine was built with an implicit batch dimension. This is an engine-wide property. Either all tensors in the engine have an implicit batch dimension or none of them do. This is True if and only if the INetworkDefinition from which this engine was built was created without the NetworkDefinitionCreationFlag.EXPLICIT_BATCH flag.

# num_layers – int The number of layers in the network. The number of layers in the network is not necessarily the number in the original INetworkDefinition, as layers may be combined or eliminated as the ICudaEngine is optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.

# max_workspace_size – int The amount of workspace the ICudaEngine uses. The workspace size will be no greater than the value provided to the Builder when the ICudaEngine was built, and will typically be smaller. Workspace will be allocated for each IExecutionContext .

# device_memory_size – int The amount of device memory required by an IExecutionContext .

# refittable – bool Whether the engine can be refit.

# name – str The name of the network associated with the engine. The name is set during network creation and is retrieved after building or deserialization.

# num_optimization_profiles – int The number of optimization profiles defined for this engine. This is always at least 1.

# error_recorder – IErrorRecorder Application-implemented error reporting interface for TensorRT objects.

# engine_capability – EngineCapability The engine capability. See EngineCapability for details.

# tactic_sources – int The tactic sources required by this engine.

# profiling_verbosity – The profiling verbosity the builder config was set to when the engine was built.

# hardware_compatibility_level – The hardware compatibility level of the engine.

# num_aux_streams – Read-only. The number of auxiliary streams used by this engine, which will be less than or equal to the maximum allowed number of auxiliary streams by setting builder_config.max_aux_streams when the engine is built.

class TensorRTBackend(object):
    def __init__(self, gpus=[0]):
        self._bindings = None
        self.devices = [cuda.Device(gpu) for gpu in gpus]
        print(f'Server has been initialized to use GPU({gpus})')
        # self.device_contexts = [device.make_context() for device in self.devices]
        self.engines = dict()
        self.contexts = dict()
        self.inputs = dict()
        self.outputs = dict()
        self.allocations = dict()
        
    @property
    def bindings(self):
        return self._bindings

    @property
    def bindings(self, value):
        self._bindings = value

    def load_model(self, model):
        model_id = model['id']
        model_name = model['name']
        model_path = model['path']
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(model_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            self.engines[model_name] = engine
        
        context = self.engines[model_name].create_execution_context()
        self.contexts[model_name] = context
        assert self.engines[model_name]
        assert self.contexts[model_name]
        
        inputs, outputs = self.alloc_buf(model_name, self.engines[model_name], self.contexts[model_name])
        return ModelDescriptor(
            name=model_name,
            id=model_id,
            num_ops=len(inputs) + len(outputs),
            num_tensors=len(inputs) + len(outputs),
            tensor_types=[],
            input_tensor_indices=[input['index'] for input in inputs],
            output_tensor_indices=[output['index'] for output in outputs],
            op_input_tensors=[],
            op_output_tensors=[],
        )

    def inference(self, model_name, input_image):
        cuda.memcpy_htod(self.inputs[model_name][0]['allocation'], input_image)
        self.contexts[model_name].execute_v2(self.allocations[model_name])
        for i in range(len(self.outputs[model_name])):
            cuda.memcpy_dtoh(self.outputs[model_name][i]['host_allocation'], self.outputs[model_name][i]['allocation'])
        # TODO

    def alloc_buf(self, model_name, engine, context):
        self.inputs[model_name] = []
        self.outputs[model_name] = []
        self.allocations[model_name] = []

        for i in range(engine.num_bindings):
            is_input = False
            if engine.binding_is_input(i):
                is_input = True
            name = engine.get_binding_name(i)
            dtype = engine.get_binding_dtype(i)
            shape = context.get_binding_shape(i)

            if is_input:
                if shape[0] < 0:
                    assert engine.num_optimization_profiles > 0
                    profile_shape = engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3
                    context.set_binding_shape(i, profile_shape[2])
                    shape = context.get_binding_shape(i)
            size = get_bytes_size(dtype)
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, trt.nptype(dtype))
            binding = {
                "index": i,
                "name": name,
                "dtype": convert_dtype(dtype),
                "shape": convert_shape(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations[model_name].append(allocation)
            if engine.binding_is_input(i):
                self.inputs[model_name].append(binding)
            else:
                self.outputs[model_name].append(binding)

        assert len(self.inputs[model_name]) > 0
        assert len(self.outputs[model_name]) > 0
        assert len(self.allocations[model_name]) > 0
        return [self.inputs[model_name], self.outputs[model_name]]

    def input_spec(self, model_name):
        specs = []
        for i in self.inputs[model_name]:
            specs.append((i['shape'], i['dtype']))
        return specs

    def output_spec(self, model_name):
        specs = []
        for o in self.outputs[model_name]:
            specs.append((o['shape'], o['dtype']))
        return specs
