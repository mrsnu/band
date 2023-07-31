#include <jni.h>

#include "band/buffer/image_processor.h"
#include "band/java/src/main/native/jni_utils.h"

using band::Buffer;
using band::BufferProcessor;
using band::Tensor;
using band::jni::JNIImageProcessor;
using namespace band::jni;

extern "C" {

BufferProcessor* ConvertJobjectToBufferProcessor(JNIEnv* env,
                                                 jobject processor) {
  JNI_DEFINE_CLS_AND_MTD(proc, "org/mrsnu/band/ImageProcessor",
                         "getNativeHandle", "()J");
  JNIImageProcessor* imageProcessor = reinterpret_cast<JNIImageProcessor*>(
      env->CallLongMethod(processor, proc_mtd));
  return imageProcessor->impl.get();
}

Buffer* ConvertJobjectToBuffer(JNIEnv* env, jobject buffer) {
  JNI_DEFINE_CLS_AND_MTD(buf, "org/mrsnu/band/Buffer", "getNativeHandle",
                         "()J");
  Buffer* buf_ptr =
      reinterpret_cast<Buffer*>(env->CallLongMethod(buffer, buf_mtd));
  return buf_ptr;
}

Tensor* ConvertJobjectToTensor(JNIEnv* env, jobject tensor) {
  JNI_DEFINE_CLS_AND_MTD(tsr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  Tensor* tsr_ptr =
      reinterpret_cast<Tensor*>(env->CallLongMethod(tensor, tsr_mtd));
  return tsr_ptr;
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_ImageProcessor_process(
    JNIEnv* env, jclass clazz, jobject imageProcessorHandle,
    jobject bufferHandle, jobject outputTensorHandle) {
  BufferProcessor* processor =
      ConvertJobjectToBufferProcessor(env, imageProcessorHandle);
  Buffer* buffer = ConvertJobjectToBuffer(env, bufferHandle);
  Tensor* outputTensor = ConvertJobjectToTensor(env, outputTensorHandle);
  std::shared_ptr<Buffer> outputTensorBuffer(
      Buffer::CreateFromTensor(outputTensor));

  auto status = processor->Process(*buffer, *outputTensorBuffer);
  if (!status.ok()) {
    // TODO: refactor absl
    return;
  }
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_ImageProcessor_deleteImageProcessor(
    JNIEnv* env, jclass clazz, jlong imageProcessorHandle) {
  delete reinterpret_cast<JNIImageProcessor*>(imageProcessorHandle);
}
}  // extern "C"