import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np

class TensorRTBackend:
    def __init__(self, gpu=None):
        self._bindings = None
        if gpu is None:
            self.device = cuda.Device(0)
        else:
            self.device = cuda.Device(gpu)
            print(f'Server has been initialized to use GPU({gpu})')
        self.ctx = self.device.make_context()
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

    def load_model(self, model_name, model_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
        runtime = trt.Runtime(TRT_LOGGER)
        with open(model_path, 'rb') as f:
            self.engines[model_name] = runtime.deserialize_cuda_engine(f.read())
        self.contexts[model_name] = self.engines[model_name].create_execution_context()
        assert self.engines[model_name]
        assert self.contexts[model_name]
        self.alloc_buf(model_name, self.engines[model_name], self.contexts[model_name])

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
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            shape = context.get_binding_shape(i)

            if is_input:
                if shape[0] < 0:
                    assert engine.num_optimization_profiles > 0
                    profile_shape = engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3
                    context.set_binding_shape(i, profile_shape[2])
                    shape = context.get_binding_shape(i)
            size = dtype.itemsize
            for s in shape:
                size *= s

            allocation = cuda.mem_alloc(size)
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations[model_name].append(allocation)
            if engine.binding_is_input(i):
                self.inputs[model_name].append(binding)
            else:
                self.outputs[model_name].append(binding)
            
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))        

        assert len(self.inputs[model_name]) > 0
        assert len(self.outputs[model_name]) > 0
        assert len(self.allocations[model_name]) > 0

    def input_spec(self, model_name):
        return self.inputs[model_name][0]['shape'], self.inputs[model_name][0]['dtype']

    def output_spec(self, model_name):
        specs = []
        for o in self.outputs[model_name]:
            specs.append((o['shape'], o['dtype']))
        return specs
