import io
import argparse
import grpc
from PIL import Image
from proto.request_pb2 import Request
from proto.response_pb2 import Response
from proto.service_pb2_grpc import BandServiceStub

def preprocess_image(image_path):
    image = Image.open(image_path)
    b = io.BytesIO()
    image.save(b, 'png')
    return b.getvalue()

def run(model, height, width, data, port=50051):
    
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        stub = BandServiceStub(channel)
        response = stub.RequestInference(
            Request(
                model=model,
                height=height,
                width=width,
                data=data,
            )            
        )
        print(f"computation time: {response.computation_time_ms} us")
        print(f"result: {response.result}")

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=str, required=True)
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-y', '--height', type=int, required=True)
parser.add_argument('-x', '--width', type=int, required=True)
parser.add_argument('-i', '--image-path', type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    image_data = preprocess_image(args.image_path)
    run(args.model, args.height, args.width, image_data, args.port)
