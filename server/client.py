from __future__ import print_function

import time
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

gpu_runtime_list = []
e2e_time_list = []

def summary():
    gpu_runtime_list_tmp = [int(runtime_string.split(" ")[3]) for runtime_string in gpu_runtime_list]
    print(f"GPU Runtime: ")
    print(f"    Avg: {sum(gpu_runtime_list_tmp) / len(gpu_runtime_list_tmp)} ms")
    print(f"    Min: {min(gpu_runtime_list_tmp)} ms")
    print(f"    Max: {max(gpu_runtime_list_tmp)} ms")
    print(f"")
    print(f"E2E Time: ")
    print(f"    Avg: {sum(e2e_time_list) / len(e2e_time_list) * 1000} ms")
    print(f"    Min: {min(e2e_time_list) * 1000} ms")
    print(f"    Max: {max(e2e_time_list) * 1000} ms")

def run():
    with grpc.insecure_channel('12.elsa.snuspl.snu.ac.kr:5005') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(helloworld_pb2.HelloRequest(name='you'))
    print("Greeter client received: " + response.message)
    gpu_runtime_list.append(response.message)

logging.basicConfig()
for i in range(1000):
    start = time.time()
    run()
    e2e_time_list.append(time.time() - start)

summary()
