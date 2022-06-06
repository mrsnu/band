from concurrent import futures
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

import tensorflow as tf

from tensorflow.keras.applications.mobilenet import MobileNet

class Greeter(helloworld_pb2_grpc.GreeterServicer):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def SayHello(self, request, context):
    with tf.device('/GPU:0'):
      x = tf.random.uniform(shape=[1,224,224,3])
      y = self.model(x)
    return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)

def serve():
  with tf.device('/GPU:0'):
    model = MobileNet(input_shape=(224,224,3))

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(model), server)
  server.add_insecure_port('[::]:50051')
  server.start()
  server.wait_for_termination()

if __name__ == '__main__':
  logging.basicConfig()
  serve()
