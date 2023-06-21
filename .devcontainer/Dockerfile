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