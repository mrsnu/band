#include <jni.h>

#include "band/config.h"
#include "band/java/src/main/native/jni_utils.h"

using band::RuntimeConfig;
using band::jni::JNIRuntimeConfig;

extern "C" {

JNIEXPORT void JNICALL Java_org_mrsnu_band_Config_deleteConfig(
    JNIEnv* env, jclass clazz, jlong configHandle) {
  delete reinterpret_cast<JNIRuntimeConfig*>(configHandle);
}

}  // extern "C"