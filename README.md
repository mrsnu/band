<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h2 align="center">Band: Multi-DNN Framework for Mobile-Cloud Platform </h2>
</div>

## Introduction

[Band](https://dl.acm.org/doi/10.1145/3498361.3538948) is an efficient deep learning platform for mobile-cloud collaborative support for multiple DNNs. 
Band supports backend-agnostic coordination of DNN requests on heterogeneous processors in a mobile device to <s>cloud server</s>.
Band is currently backed by following backend machine learning frameworks.

|         | [Tensorflow v2.9.2](https://github.com/tensorflow/tensorflow/tree/v2.9.2)    | ... |
|---------|--------------------|-----|
| Android |  &#9745; |     |
| iOS     |  &#9744; |     |
| gRPC    |  &#9744; |     |

Band provides Java and C APIs, as well as an official plugin for [Unreal Engine](https://www.unrealengine.com/).

## Useful Links

* Examples (requires update)
* [Unreal Engine Plugin](https://github.com/mrsnu/ue4-plugin)

## Codebase

```
.
├── band
│   ├── backend  # backend-specific implementation of `interface`
│   ├── c  # C API
│   ├── docs
│   ├── interface  # Backend-agnostic interfaces. Each backend (e.g., Tensorflow Lite, MNN, ...) should implement them to communicate with Band core
│   ├── java  # Java API
│   ├── scheduler  # Schedulers
│   ├── test  # Test codes
│   ├── tool  # Benchmark tools
├── script  # Utilities
├── third_party
└── WORKSPACE
```

## Getting Started

### Prerequisites

* Install Android SDK 28, NDK v19.2.53456 
    * or create [Visual Studio Code Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) using `[root]/.devcontainer`
    * or utilize `[root]/.devcontainer/Dockerfile`

* Configure Android SDK, NDK for build system (Bazel)
  ```sh
  python configure.py
  ```

### How to Build / Run

Refer to detailed instructions in `[root]/script`

* Run test 
  ```sh
  python script/run_test.py -android 
  ```

* Build Android AAR
  ```sh
  sh script/build_aar_armv8.sh
  ```

* Build C API
  ```sh
  python script/build_c_api.py -android 
  ```

* Run benchmark -- check `[root]/docs/benchmark.md`

## Citation

If you find our work useful, please cite our paper below!
The original codebase for paper submission is archived [here](https://github.com/mrsnu/band/releases/tag/v0.0.0)
```
@inproceedings{jeong2022band,
  title={Band: coordinated multi-DNN inference on heterogeneous mobile processors},
  author={Jeong, Joo Seong and Lee, Jingyu and Kim, Donghyun and Jeon, Changmin and Jeong, Changjin and Lee, Youngki and Chun, Byung-Gon},
  booktitle={Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services},
  pages={235--247},
  year={2022}
}
```

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Tensorflow](https://github.com/tensorflow/tensorflow)
* [NCNN](https://github.com/Tencent/ncnn) - CPU affinity control

<p align="right">(<a href="#top">back to top</a>)</p>
