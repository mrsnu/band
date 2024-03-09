#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT JNICALL void Java_org_mrsnu_band_Band_nativeDoNothing(
    JNIEnv* env, jclass /*clazz*/) {
  // Do nothing. Used for check if the native library is loaded.
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus