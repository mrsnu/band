# Inference Server for Offloading DNNs

## Preparing Models
### Downloading models

You can download models keras supports by running `util/keras_models.py`.  
Current suppoorted models are:
- MobileNetV1
- MobileNetV2
- MobileNetV3-Small
- MobileNetV3-Large
- ResNet50

Download and save the models by running:
```sh
python util/keras_models.py
```
With `--model` argument, you can specify the model that you want to save. By default, it will download all the model supported.

### Converting a model to a TensorRT model
To convert a model to support TensorRT model, `util/trt_convert.py` is provided.

```sh
python util/trt_convert.py --input-model-dir=<model-dir> --output-model-dir=<output-dir>
```

With the `--input-model-dir` argument, you can specify the model that should be converted. The `--output-model-dir` specifies the output directory of the converted model. 


### Putting all together
We provides an all-in-one bash script to download models and convert them into TensorRT models. Just run
```sh
bash util/download_and_convert.sh
```


