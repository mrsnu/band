from concurrent import futures
import logging
import os
import grpc

import helloworld_pb2
import helloworld_pb2_grpc

import datetime
import tensorflow as tf
import numpy as np



def get_filepath(filename, extension):
    return f'{filename}{extension}'

class Greeter(helloworld_pb2_grpc.GreeterServicer):

    def __init__(self):
        self.last_time = datetime.datetime.now()

    def SayHello(self, request, context):
        return helloworld_pb2.StringResponse(message=f'Hello, {request.name}! Your age is {request.age}')

    def UploadFile(self, request_iterator, context):
        data = bytearray()
        filepath = './dummy.csv'
        count = 0

        for request in request_iterator:
            # time_diff = (datetime.datetime.now() - self.last_time)
            # execution_time = time_diff.total_seconds() * 1000 # in miliseconds
            # print("{time} ms".format(time=execution_time))
            # self.last_time = datetime.datetime.now()
            filepath = get_filepath(request.metadata.filename, request.metadata.extension)
            data.extend(request.chunk_data)
            print('upload starts = ' + str(count))
            count = count + 1
        with open(filepath, mode="wb") as f:
            f.write(data)
        return helloworld_pb2.StringResponse(message='Success!')

    def DownloadFile(self, request, context):
        print('download starts')
        chunk_size = 102400
        count = 0

        filepath = f'{request.filename}{request.extension}'
        if os.path.exists(filepath):
            with open(filepath, mode="rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    print('download read = ' + str(count))
                    count = count + 1 
                    if chunk:
                        entry_response = helloworld_pb2.FileResponse(chunk_data=chunk)
                        yield entry_response
                    else:  # The chunk was empty, which means we're at the end of the file
                        return
        else:
            print('file not exists')
            


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:5005')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()