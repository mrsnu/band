import argparse
from concurrent import futures
import multiprocessing
import time
import json
import grpc
import numpy as np
from backend.tensorrt import TensorRTBackend
from config import get_config

from proto import *
from proto.service_pb2_grpc import (
    BandServiceServicer,
    add_BandServiceServicer_to_server
)

SERVERS = []

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="Config json file.")
args = parser.parse_args()


class InferenceServer(BandServiceServicer):
    def __init__(
            self,
            models,
            backend="tensorrt",
            gpus=[]):
        self.backend = TensorRTBackend(gpu)
        self.model_descs = []
        for model in models:
            self.model_descs.append(self.backend.load_model(model))

    def GetModelDesc(self, request, context):
        for model in self.model_descs:
            yield model

    def CheckModelDesc(self, request, context):
        return Status(code=StatusCode.OK) if request in self.model_descs else Status(code=StatusCode.MODEL_NOT_FOUND)


def main():
    port, models = get_config(args.config)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    SERVERS.append(server)
    add_BandServiceServicer_to_server(InferenceServer(models), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f'Server started on port {port}')
    server.wait_for_termination()


if __name__ == "__main__":
    main()
