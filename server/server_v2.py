import argparse
from concurrent import futures
import datetime
import logging

import tensorflow as tf
import numpy as np

from random import randint
from threading import Thread

import grpc
import helloworld_pb2
import helloworld_pb2_grpc


parser = argparse.ArgumentParser()
parser.add_argument("--workload", type=str, default=None, required=False)
args = parser.parse_args()

try:
    workload_list = args.workload.split(",")
except:
    workload_list = [
            "MobileNetV1", 
            "MobileNetV1TRT",
            "MobileNetV2",
            "MobileNetV2TRT",
            "MobileNetV3Small",
            "MobileNetV3SmallTRT",
            "MobileNetV3Large",
            "MobileNetV3LargeTRT",
            "ResNet50",
            "ResNet50TRT"
        ]

MobileNetV1 = {
        "file_path": "./models/mobilenetv1",
        "shape": (1, 224, 224, 3),
        "name": "MobileNet"
    }

MobileNetV1TRT = {
        "file_path": "./models/mobilenetv1_trt",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetTRT"
    }

MobileNetV2 = {
        "file_path": "./models/mobilenetv2",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV2"
    }

MobileNetV2TRT = {
        "file_path": "./models/mobilenetv2_trt",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV2TRT"
    }

MobileNetV3Small = {
        "file_path": "./models/mobilenetv3-small",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV3Small"
    }

MobileNetV3SmallTRT = {
        "file_path": "./models/mobilenetv3-small_trt",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV3SmallTRT"
    }

MobileNetV3Large = {
        "file_path": "./models/mobilenetv3-large",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV3Large"
    }

MobileNetV3LargeTRT = {
        "file_path": "./models/mobilenetv3-large_trt",
        "shape": (1, 224, 224, 3),
        "name": "MobileNetV3LargeTRT"
    }

ResNet50 = {
        "file_path": "./models/resnet50",
        "shape": (1, 224, 224, 3),
        "name": "ResNet50"
    }

ResNet50TRT = {
        "file_path": "./models/resnet50_trt",
        "shape": (1, 224, 224, 3),
        "name": "ResNet50TRT"
    }

class Job(object):
    def __init__(self, model, following, batch_size):
        self.model = model
        self.following = following
        self.batch_size = batch_size
        self.thread_list = list()

    def AddFollowing(self, job):
        self.following.append(job)

    def Run(self):
        for i in range(0, self.batch_size):
            self.model.Run()
        for following_model in self.following:
            following_model.Run()

class Model(object):
    def __init__(self, config):
        self.config = config
        self.model = None
        with tf.device('/GPU:0'):
            # Load saved_model
            self.model = tf.saved_model.load(config['file_path'])
            self.x = np.ndarray(shape=self.config['shape'], dtype=np.float32)
    
    def Run(self):
        start_time = datetime.datetime.now()
        with tf.device('/GPU:0'):
            predictions = self.model(self.x)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        print(f"{self.config['name']} = {execution_time}")

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def __init__(self):
        super().__init__()
        self.workload = list()
        
        for workload_name in workload_list:
            setattr(self, workload_name, Model(globals()[workload_name]))
            self.workload += [Job(getattr(self, workload_name), list(), 1)] * 10
            

    def SayHello(self, request, context):
        start_time = datetime.datetime.now()
        for work in self.workload:
            work.Run()

        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds() * 1000
        return helloworld_pb2.HelloReply(
                message=f"Done, {request.name}! : {int(execution_time)} ms"
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:5005')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    serve()
