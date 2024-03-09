mkdir /android

wget https://dl.google.com/android/repository/commandlinetools-linux-6200805_latest.zip
unzip commandlinetools-linux-6200805_latest.zip
rm commandlinetools-linux-6200805_latest.zip

yes | /android/tools/bin/sdkmanager --licenses --sdk_root=/android
/android/tools/bin/sdkmanager --sdk_root=android "platforms;android-28" "build-tools;28.0.3" "ndk;18.1.5063045"

echo "export ANDROID_DEV_HOME=$HOME/android" >> ~/.bashrc
echo "export ANDROID_API_LEVEL=28" >> ~/.bashrc
echo "export ANDROID_NDK_API_LEVEL=21" >> ~/.bashrc

echo "export ANDROID_BUILD_TOOLS_VERSION=28.0.3" >> ~/.bashrc
echo "export ANDROID_SDK_HOME=$ANDROID_DEV_HOME" >> ~/.bashrc

echo "export ANDROID_NDK_HOME=$ANDROID_DEV_HOME/ndk/18.1.5063045" >> ~/.bashrc

source ~/.bashrc

PATH=$PATH:$ANDROID_SDK_HOME/tools:$ANDROID_SDK_HOME/platform-tools:$ANDROID_NDK_HOME

chmod -R go=u $ANDROID_DEV_HOME