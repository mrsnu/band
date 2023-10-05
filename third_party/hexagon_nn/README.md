# Hexagon libraries for TensorFlow Lite Hexagon delegate

* `libhexagon_interface.so` - Hexagon NN interface [v1.20](https://storage.googleapis.com/mirror.tensorflow.org/storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_headers_v1.20.0.1.tgz)
* `libhexagon_nn_skel_*.so` - Hexagon NN skeleton library [v1.20](https://storage.cloud.google.com/download.tensorflow.org/tflite/hexagon_nn_skel_v1.20.0.1.run)
  

This directory contains the prebuilt Hexagon libraries for the TensorFlow Lite.
This libraries are intended to be optionally used by the TensorFlow Lite backend when Qualcomm Hexagon DSP access is not available from the NNAPI (e.g., either direct access is prohibited or simply does not have NNAPI DSP driver).
You need to accept the license agreement before using these libraries.

Refer to the [Hexagon Delegate](https://www.tensorflow.org/lite/android/delegates/hexagon) for more information.

