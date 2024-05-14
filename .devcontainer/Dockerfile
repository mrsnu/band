# Copyright 2023 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM tensorflow/tensorflow:devel-gpu

ENV ANDROID_DEV_HOME /android
RUN mkdir -p ${ANDROID_DEV_HOME}

# Install Android SDK.
ENV ANDROID_CMD_TOOLS_FILENAME commandlinetools-linux-6200805_latest.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/
ENV ANDROID_API_LEVEL 28
ENV ANDROID_NDK_API_LEVEL 21

# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION 30.0.0
# r19c
ENV ANDROID_NDK_VERSION 19.2.5345600 
ENV ANDROID_SDK_HOME ${ANDROID_DEV_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL}${ANDROID_CMD_TOOLS_FILENAME} && \
    unzip ${ANDROID_CMD_TOOLS_FILENAME} && \
    rm ${ANDROID_CMD_TOOLS_FILENAME}
RUN yes | ${ANDROID_DEV_HOME}/tools/bin/sdkmanager --licenses --sdk_root=${ANDROID_DEV_HOME}
RUN ${ANDROID_DEV_HOME}/tools/bin/sdkmanager --sdk_root=${ANDROID_DEV_HOME} "platforms;android-${ANDROID_API_LEVEL}" "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" "ndk;${ANDROID_NDK_VERSION}"

ENV ANDROID_NDK_HOME ${ANDROID_DEV_HOME}/ndk/${ANDROID_NDK_VERSION}
ENV PATH ${PATH}:$ANDROID_SDK_HOME/tools:$ANDROID_SDK_HOME/platform-tools:$ANDROID_NDK_HOME

# Make android ndk executable to all users.
RUN chmod -R go=u ${ANDROID_DEV_HOME}

# OpenCL
# ref : https://stackoverflow.com/questions/54243119/how-to-run-opencl-opengl-inside-a-docker-container
RUN apt-get update && apt-get install -y \
    ocl-icd-opencl-dev \
    clinfo &&\
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# # For TFLite build with OpenCL-NVIDIA GPU
# RUN apt-get update && apt-get install -y cmake 
# RUN mkdir /tflite_build && cd /tflite_build
# # RUN cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Debug
# RUN cmake ../tensorflow_src/tensorflow/lite -DTFLITE_ENABLE_GPU=ON
# RUN cmake --build . -j

# For Debugging
RUN apt-get update && apt-get install -y gdb

# Build essentials
RUN apt-get update && apt-get install -y cmake git vim wget

# fzf settings
RUN git clone https://github.com/junegunn/fzf.git ~/.fzf
RUN ~/.fzf/install
RUN source ~/.bashrc