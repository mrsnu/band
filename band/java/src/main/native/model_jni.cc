#include <jni.h>

#include "band/java/src/main/native/jni_utils.h"
#include "band/model.h"

using Band::Model;
using Band::jni::convertLongToModel;

namespace {

jobject convertCcSetToJavaSet(JNIEnv* env, std::set<BandBackendType> cc_set) {
  jclass set_class = env->FindClass("java/util/HashSet");
  jmethodID set_ctor = env->GetMethodID(set_class, "<init>", "()V");
  jmethodID set_add_method = env->GetMethodID(set_class, "add", "(Ljava/lang/Object;)Z");
  jmethodID set_size_method = env->GetMethodID(set_class, "size", "()I");

  jclass backend_type_class = env->FindClass("org/mrsnu/band/BackendType");
  jmethodID backend_type_ctor = env->GetMethodID(backend_type_class, "<init>", "(I)V");

  jobject java_set = env->NewObject(set_class, set_ctor);
  for (auto elem : cc_set) {
    jobject java_backend_type = env->NewObject(backend_type_class, backend_type_ctor);
    env->CallBooleanMethod(java_set, set_add_method, java_backend_type);
  }
  return java_set;
}
  
}  // anonymouse namepsace

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeModelWrapper_createModel(JNIEnv* env, jclass clazz) {
  return reinterpret_cast<jlong>(new Model());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeModelWrapper_delete(
    JNIEnv* env, jclass clazz, jlong model_handle) {
  Model* model = convertLongToModel(env, model_handle);
  delete model;
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeModelWrapper_loadFromFile(
    JNIEnv* env, jclass clazz, jlong model_handle, jint backend_type,
    jstring file_path) {
  Model* model = convertLongToModel(env, model_handle);
  model->FromPath((BandBackendType)backend_type,
                  env->GetStringUTFChars(file_path, nullptr));
}

JNIEXPORT jobject JNICALL
Java_org_mrsnu_band_NativeModelWrapper_getSupportedBackends(JNIEnv* env, jclass clazz, jlong model_handle) {
  Model* model = convertLongToModel(env, model_handle);
  return convertCcSetToJavaSet(env, model->GetSupportedBackends());
}
