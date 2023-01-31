<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h2 align="center">Band: Multi-DNN Framework for Mobile-Cloud Platform </h2>
</div>

### Introduction

[Band](https://dl.acm.org/doi/10.1145/3498361.3538948) is an efficient deep learning platform for mobile-cloud collaborative support for multiple DNNs. It supports coordination of multi-DNN requests on heterogeneous processors in mobile device to cloud GPU. Band is backed by following backend frameworks.

<!-- Add more .. -->
* Tensorflow Lite


### Codebase

```
[root]/band
├── backend 
├── c
├── docs
├── interface
├── java
├── scheduler
├── test
└── testdata
    └── testdata    
[root]/tensorflow <Same as TF v2.9.2>
```
* Tensorflow Lite in [Tensorflow v2.9.2](https://github.com/tensorflow/tensorflow/tree/v2.9.2) 
* Among files and directories located in `[root]/tensorflow`, files unrelated to TF Lite are deleted.
  * The following directories are necessary to build `benchmark tool` and `Android library`.




<!-- GETTING STARTED -->
### Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
   
### How to Run

Run below script to start the app
```sh
bash run.sh
```

<p align="right">(<a href="#top">back to top</a>)</p>


### Citation

If you find our work useful, please cite our paper below!
```
@inproceedings{jeong2022band,
  title={Band: coordinated multi-DNN inference on heterogeneous mobile processors},
  author={Jeong, Joo Seong and Lee, Jingyu and Kim, Donghyun and Jeon, Changmin and Jeong, Changjin and Lee, Youngki and Chun, Byung-Gon},
  booktitle={Proceedings of the 20th Annual International Conference on Mobile Systems, Applications and Services},
  pages={235--247},
  year={2022}
}
```


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
### Acknowledgments

* [Tensorflow](https://github.com/tensorflow/tensorflow)

<p align="right">(<a href="#top">back to top</a>)</p>
