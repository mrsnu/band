#include <jni.h>

#include "band/java/src/main/native/jni_utils.h"
#include "band/tensor.h"

using Band::Tensor;
using Band::jni::ConvertLongToTensor;

JNIEXPORT jint JNICALL
Java_org_mrsnu_band_TensorImpl_getType(JNIEnv* env, jclass clazz, jlong tensor_jandle) {
  
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_TensorImpl_setType(JNIEnv* env, jclass clazz, jlong tensor_handle, int data_type) {
  
}

JNIEXPORT jbyteArray JNICALL
Java_org_mrsnu_band_TensorImpl_getDims(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {

}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_TensorImpl_setDims(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {

}

JNIEXPORT jstring JNICALL
Java_org_mrsnu_band_TensorImpl_getName(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {

}

JNIEXPORT jint JNICALL
Java_org_mrsnu_band_TensorImpl_getQuantizationType(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {

}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_TensorImpl_getQUantizationParams(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {
  
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_TensorImpl_setQuantization(JNIEnv* env, jclass clazz, jlong tensor_handle, ) {

}