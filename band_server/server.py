from concurrent import futures
import multiprocessing
import time
import grpc
import numpy as np
from backend.tensorrt import TensorRTBackend

from proto.splash_pb2 import (
    Request,
    Response
)
from proto.splash_pb2_grpc import (
    SplashServicer,
    add_SplashServicer_to_server
)

SERVERS = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=False, default=50051)
args = parser.parse_args() 

MODEL_CONFIG = {
    "image_classification": {
        "height": 224,
        "width": 224,
        "model_path": "splash/models/mobilenetv2.trt"
    },
    "text_detection": {
        "height": 512,
        "width": 512,
        "model_path": "splash/models/east.trt"
    },
    "face_detection": {
        "height": 160,
        "width": 160,
        "model_path": "splash/models/retinaface-mbv2.trt"
    },
    "face_recognition": {
        "height": 112,
        "width": 112,
        "model_path": "splash/models/arcface-mbv2.trt",
    },
    "object_detection": {
        "height": 320,
        "width": 320,
        "model_path": "splash/models/efficientdet-lite0.trt"
    }
}

class InferenceServer(SplashServicer):
    def __init__(self):
        self.__initialized = False
        self.backend = None

    def RequestInference(self, request, context):
        print("request comes = {}".format(request.model))
        if request.model == "nothing":
            return Response(
                    computation_time_ms=0,
                    result=None)
        if request.model == "image_classification":
            now = time.time()
            time.sleep(670/1000/1000)
            return Response(
                    computation_time_ms=int((time.time() - now) * 1000000),
                    result=None)
                    
        model_config = MODEL_CONFIG.get(request.model)

        if self.__initialized == False:
            self.backend = TensorRTBackend(gpu=args.gpu)
            for model_name in MODEL_CONFIG:
                self.backend.load_model(model_name, MODEL_CONFIG[model_name]["model_path"])
            self.__initialized = True
            
        now = time.time()
        # dummy input for now
        self.run_model(request.model, np.array(
            [
                1,  # batch size
                request.height, 
                request.width, 
                3  # channels
            ], 
            dtype=np.int8
        ))
        print("compute time = {}".format(int((time.time() - now) * 1000000)))
        return Response(
            computation_time_ms=int((time.time() - now) * 1000000),
            result=None)
    
    def run_model(self, model, input_image):
        return self.backend.inference(model, input_image)

def start_server(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    SERVERS.append(server)
    add_SplashServicer_to_server(InferenceServer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()

if __name__ == "__main__":
    start_server()
