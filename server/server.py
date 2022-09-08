from concurrent import futures
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

import datetime
import tensorflow as tf
import numpy as np

from random import randint
from threading import Thread

from tensorflow.python.platform import gfile
from tensorflow.keras.applications.mobilenet import MobileNet as MN

# CAUTION : Download models from google drive into ./models/
ICN = {
    "file_path": "./models/ICN.pb",
    "input_node": "input_LR_patch:0",
    "output_layer": "generator_decoder_fine/CONV_Out/BiasAdd:0",
    "shape": (1,14,14,3),
    "name": "ICN",
}
MobileNet = {
    "file_path": "./models/ArcFaceMobileNet.pb",
    "input_node": "input_place:0",
    "output_layer": "mobileFaceNet/output:0",
    "shape": (1,112,112,3),
    "name": "MobileNet",
}
ResNet = {
    "file_path": "./models/ArcFaceResNet50.pb",
    "input_node": "input_place_1:0",
    "output_layer": "resnet50/output:0",
    "shape": (1,112,112,3),
    "name": "ResNet",
}
RetinaFace = {
    "file_path": "./models/RetinaFaceMobileNet-1080x1920.pb",
    "input_node": "data:0",
    "output_layer": "face_rpn_landmark_pred_stride8_rev:0",
    "shape": (1,1080,1920,3),
    "name": "RetinaFace",
}

class Job():
  def __init__(self, model, following, batchSize):
    self.model = model
    self.following = following
    self.batchSize = batchSize
    self.threadList = list()

  def AddFollowing(self, job):
    self.following.append(job)

  def Run(self):
    for i in range(0, self.batchSize):
      self.model.Run()
    for i in range(0, len(self.following)):
      self.following[i].Run() # TODO : MultiThreading

class Model():
  def __init__(self, model):
    self.model = model
    with tf.device('/GPU:0'):
      self.sess = tf.compat.v1.Session()
      with gfile.FastGFile(model['file_path'],'rb') as f:
        self.graph_def = tf.compat.v1.GraphDef()
        self.graph_def.ParseFromString(f.read())
      self.sess.graph.as_default()
      tf.import_graph_def(self.graph_def, name='')
      self.x = np.ndarray(shape=self.model['shape'])
      self.prob_tensor = self.sess.graph.get_tensor_by_name(self.model['output_layer'])

  def Run(self):
    start_time = datetime.datetime.now()
    with tf.device('/GPU:0'):
      predictions = self.sess.run(self.prob_tensor, {self.model['input_node']: self.x})
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000 # in miliseconds
    print('{model} = {time}'.format(model=self.model['name'], time=execution_time))

class Greeter(helloworld_pb2_grpc.GreeterServicer):
  def __init__(self):
    super().__init__()
    self.RetinaFace = Model(RetinaFace)
    self.MobileNet = Model(MobileNet)
    self.ResNet = Model(ResNet)
    self.ICN = Model(ICN)

    self.workload = list()

    for i in range(0, 8):
      self.workload.append(Job(self.RetinaFace, list(), 1))

    for i in range(0, 4):
      self.workload[randint(0, 7)].AddFollowing(Job(self.MobileNet, list(), 1))

    for i in range(0, 2):
      self.workload[randint(0, 7)].AddFollowing(Job(self.ResNet, list(), 1))

    for i in range(0, 5):
      self.workload[randint(0, 7)].AddFollowing(Job(self.ICN, [Job(self.ResNet, list(), 1)], 1))

  def SayHello(self, request, context):
    start_time = datetime.datetime.now()
    for i in range(0, len(self.workload)):
      self.workload[i].Run() # TODO : Multi-Threading

    # threadList = list()
    # for i in range(0, len(self.workload)):
    #   threadList.append(Thread(target=self.workload[i].Run))
    # for i in range(0, len(threadList)):
    #   threadList[i].start()
    # for i in range(0, len(threadList)):
    #   threadList[i].join()

    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)
    execution_time = time_diff.total_seconds() * 1000 # in miliseconds
    return helloworld_pb2.HelloReply(message="Done, {name}! : {time} ms".format(name = request.name, time = int(execution_time)))

def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
  server.add_insecure_port('[::]:5005')
  server.start()
  server.wait_for_termination()

if __name__ == '__main__':
  logging.basicConfig()
  serve()
