#include <jni.h>
#include <string.h>

#include "band/java/src/main/native/jni_utils.h"
#include "band/logger.h"
#include "band/tensor.h"

using Band::Tensor;
using Band::jni::ConvertLongToTensor;

namespace {

jobject ConvertNativeToQuantization(JNIEnv* env,
                                    BandQuantization&& quantization) {
  JNI_DEFINE_CLS_AND_MTD(quant_type,
                         "org/mrsnu/band/Quantization/QuantizationType",
                         "<init>", "(I)V");
  JNI_DEFINE_CLS_AND_MTD(quant, "org/mrsnu/band/Quantization", "<init>",
                         "(Lorg/mrsnu/band/Quantization/QuantizationType;J)V");

  jobject new_quant_type = env->NewObject(quant_type_cls, quant_type_mtd,
                                          static_cast<jint>(quantization.type));
  return env->NewObject(quant_cls, quant_mtd, new_quant_type,
                        reinterpret_cast<jlong>(quantization.params));
}

BandQuantization ConvertQuantizationToNative(JNIEnv* env,
                                             jobject quantization) {
  JNI_DEFINE_CLS(quant, "org/mrsnu/band/Quantization");
  JNI_DEFINE_MTD(quant_get_type, quant_cls, "getQuantizationType",
                 "()Lorg/mrsnu/band/Quantization/QuantizationType;");
  JNI_DEFINE_MTD(quant_get_param, quant_cls, "getParamHandle", "()J");
  JNI_DEFINE_CLS_AND_MTD(quant_type,
                         "org/mrsnu/band/Quantization/QuantizationType",
                         "getValue", "()I");
  BandQuantization ret;
  ret.type = static_cast<BandQuantizationType>(env->CallIntMethod(
      env->CallObjectMethod(quantization, quant_get_param_mtd),
      quant_type_mtd));
  ret.params = reinterpret_cast<void*>(
      env->CallLongMethod(quantization, quant_get_param_mtd));
  return ret;
}

jbyteArray ConvertNativeToByteArray(JNIEnv* env, int bytes, const char* data) {
  jbyteArray arr = env->NewByteArray(bytes);
  // Note: Maybe JNI uses `signed char` to represent bytes. It can be
  // problematic if arithmetic operations are performed on the data.
  env->SetByteArrayRegion(arr, 0, bytes,
                          reinterpret_cast<const signed char*>(data));
  return arr;
}

jintArray ConvertNativeToIntArray(JNIEnv* env, int length, const int* array) {
  jintArray arr = env->NewIntArray(length);
  env->SetIntArrayRegion(arr, 0, length, array);
  return arr;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jint JNICALL Java_org_mrsnu_band_NativeTensorWrapper_getType(
    JNIEnv* env, jclass clazz, jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return static_cast<jint>(tensor->GetType());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeTensorWrapper_setType(
    JNIEnv* env, jclass clazz, jlong tensorHandle, jint dataType) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  tensor->SetType(static_cast<BandType>(dataType));
}

JNIEXPORT jbyteArray JNICALL Java_org_mrsnu_band_NativeTensorWrapper_getData(
    JNIEnv* env, jclass clazz, jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return ConvertNativeToByteArray(env, tensor->GetBytes(), tensor->GetData());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeTensorWrapper_setData(
    JNIEnv* env, jclass clazz, jlong tensorHandle, jbyteArray data) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  void* tensor_buffer = tensor->GetData();
  memcpy(tensor_buffer,
         static_cast<void*>(env->GetByteArrayElements(data, NULL)),
         tensor->GetBytes());
}

JNIEXPORT jintArray JNICALL Java_org_mrsnu_band_NativeTensorWrapper_getDims(
    JNIEnv* env, jclass clazz, jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return ConvertNativeToIntArray(env, tensor->GetNumDims(), tensor->GetDims());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeTensorWrapper_setDims(
    JNIEnv* env, jclass clazz, jlong tensorHandle, jintArray dims) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  jsize size = env->GetArrayLength(dims);
  std::vector<int> native_dims;
  for (int i = 0; i < size; i++) {
    native_dims.push_back(env->GetIntArrayElements(dims, nullptr)[i]);
  }
  tensor->SetDims(native_dims);
}

JNIEXPORT int JNICALL Java_org_mrsnu_band_NativeTensorWrapper_getBytes(
    JNIEnv* env, jclass clazz, jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return tensor->GetBytes();
}

JNIEXPORT jstring JNICALL Java_org_mrsnu_band_NativeTensorWrapper_getName(
    JNIEnv* env, jclass clazz, jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return env->NewStringUTF(tensor->GetName());
}

JNIEXPORT jobject JNICALL
Java_org_mrsnu_band_NativeTensorWrapper_getQuantization(JNIEnv* env,
                                                        jclass clazz,
                                                        jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  return ConvertNativeToQuantization(env, tensor->GetQuantization());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeTensorWrapper_setQuantization(
    JNIEnv* env, jclass clazz, jlong tensorHandle, jobject quantization) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  tensor->SetQuantization(ConvertQuantizationToNative(env, quantization));
}

}  // extern "C"