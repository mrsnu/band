from proto import *

class Model(object):
    def __init__(self, id, name, model_path):
        self.name = name
        self.model_path = model_path
        self.num_ops = 0
        self.num_tensors = 0
        self.op_input_tensors = []
        self.op_output_tensors = []
        self.tensor_types = []
        