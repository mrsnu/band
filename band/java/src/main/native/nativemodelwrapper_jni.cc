#include <jni.h>

#include "band/model.h"

using Band::Model;

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeModelWrapper_createModel(JNIEnv* env, jclass clazz) {}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeModelWrapper_deleteModel(JNIEnv* env, jclass clazz, jlong modelHandle) {
  
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeModelWrapper_loadFromFile(long modelHandle, int backendType, String filePath) {}

  //private static native List<BackendType> getSupportedBackends(long modelHandle);
}  // extern "C"